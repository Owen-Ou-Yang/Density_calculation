"""Disk-backed parent intake tests; synthetic artifacts, no model execution."""

import copy
import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from polymer_batch.catalog import load_catalog, task_seed
from thermal_properties import simulation
from thermal_properties.config import ThermalConfigError
from thermal_properties.density_parent import (
    catalog_smiles_identity, normalize_density_parent, smiles_parent_selector,
    smiles_native_run_id, verify_density_parent,
)
from thermal_properties.provenance import canonical_sha256, sha256_file
from thermal_properties.sampling_continuation import REQUIRED_CHECKS
from thermal_properties.snapshot_contract import build_snapshot_contract_payload


ROOT = Path(__file__).resolve().parents[1]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _record(path):
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def make_smiles_parent(tmp_path, *, task_index=1, attempt_number=1, parent=None):
    """Return a complete synthetic 50 ps ESS-failed parent and catalog task.

    Exported for CLI continuation tests. All inputs and runtime files are real
    temporary files; their contents are deliberately not a loadable ML model.
    """
    task = load_catalog(ROOT / "inputs/smiles.csv")[task_index - 1]
    attempt = tmp_path / task["task_id"] / f"attempt_{attempt_number:04d}"
    prep = attempt / "prep"
    prep.mkdir(parents=True)
    data = prep / "input.data"
    data.write_text("synthetic only\n\n4 atoms\n2 atom types\n"
                    "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
                    "Masses\n\n1 12.011\n2 1.008\n\nAtoms # full\n\n"
                    "1 1 1 0 1 1 1\n2 1 2 0 2 1 1\n3 1 2 0 1 2 1\n4 1 2 0 1 1 2\n")
    metadata = prep / "input.snapshot.json"
    contract = build_snapshot_contract_payload(snapshot_path=data, snapshot_class="CLASSICAL_EQ2",
        packing_id="original_packing", element_counts={"C": 1, "H": 3})
    contract["source_method"] = "RADONPY_EQ21"
    _write(metadata, contract)
    request = {"schema_version": "polymer-smiles-task/v1", **task,
        "seed": task_seed(task["smiles_sha256"]), "attempt_id": attempt.name,
        "attempt_root": str(attempt), "prepared_manifest_path": str(prep / "prepared.json")}
    if parent is not None:
        request["density_parent_restart"] = parent["selector"]
    prepared = {"task_id": task["task_id"], "smiles_sha256": task["smiles_sha256"],
        "input_data": str(data), "snapshot_metadata": str(metadata), "mace_elements": ["C", "H"],
        "preparation_provenance": {"original_smiles": task["smiles"],
            "smiles_sha256": task["smiles_sha256"], "source_method": "RADONPY_EQ21"}}
    _write(attempt / "request.json", request)
    _write(prep / "prepared.json", prepared)
    model = tmp_path / "fake_model.pt"
    model.write_bytes(b"synthetic non-model bytes")
    dependency = tmp_path / "fake_runtime.bin"
    dependency.write_bytes(b"synthetic non-runtime bytes")
    config = json.loads((ROOT / "configs/protocol.json").read_text())
    config.pop("sampling_continuation", None)  # Reproduce the immutable original single-window run.
    config["system"].update(polymer_id=task["task_id"], input_data=str(data),
        snapshot_metadata=str(metadata), mace_model=str(model))
    config["engine"].update(lammps_command=[str(dependency)], runtime_dependencies=[str(dependency)])
    config["replicas"] = [{"replica_id": "packing_001", "seed": request["seed"]}]
    if parent is not None:
        config["density_parent_restart"] = parent["selector"]
        config["initialize"]["mace_transition_npt_ps"] = 25.
    config["output_root"] = str(attempt / "density/native_runs")
    config_path = attempt / "density/mace_config.json"
    _write(config_path, config)
    resolved = simulation.resolve_thermal_config(config_path)
    identity = simulation.build_execution_identity(resolved, require_primary_executable=False)
    resolved["execution_identity"] = identity
    resolved["resolved_spec_sha256"] = simulation.resolved_spec_sha256(resolved)
    run_id = smiles_native_run_id(request, attempt / "request.json")
    run_root = Path(config["output_root"]) / (run_id + ".incomplete")
    stage = run_root / "replicas/packing_001/mace_transition_npt"
    _write(run_root / "spec/run_spec.json", resolved)
    stage_spec = {"stage_id": "packing_001__mace_transition_npt", "replica_id": "packing_001",
        "stage_role": "mace_transition_npt", "initialize_velocities": False,
        "execution_identity": identity, "start_step": 4000, "equilibration_steps": 0,
        "production_steps": 200000,
        "state": {"temperature_start_k": 305, "temperature_end_k": 305,
            "pressure_start_bar": 1.01325, "pressure_end_bar": 1.01325,
            "use_for_density": False, "use_for_tg_fit": False}}
    _write(stage / "stage_spec.json", stage_spec)
    requirement = simulation._statepoint_qc_policy_requirements(stage_spec)[0]
    qc = {"status": "FAIL", "diagnostic_observables": {"all_finite": True, "missing_columns": []},
        "qc_implementation_sha256": simulation.density_qc_implementation_sha256(identity),
        "policy_results": [{**{key: requirement[key] for key in ("role", "policy_id", "policy_sha256")},
            "status": "FAIL", "convergence": {"status": "FAIL", "checks": [
                {"name": key, "status": "FAIL" if key == "minimum_effective_samples" else "PASS",
                 "actual": 42 if key == "minimum_effective_samples" else True,
                 "criterion": ">=50" if key == "minimum_effective_samples" else True}
                for key in sorted(REQUIRED_CHECKS)]}}]}
    _write(stage / "qc/state_point_qc.json", qc)
    samples = stage / "samples/production.csv"
    samples.parent.mkdir(parents=True)
    with samples.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "time_ps", "atom_count", "volume_A3", "density_g_cm3", "lx_A", "ly_A", "lz_A"])
        writer.writerow([204000, 51, 4, 1000, 1, 10, 10, 10])
    (stage / "restart.final").write_bytes(b"synthetic completed transition restart")
    manifest = {"stage_id": stage_spec["stage_id"], "replica_id": "packing_001",
        "execution_status": "COMPLETE", "qc_status": "FAIL",
        "stage_spec_sha256": canonical_sha256(stage_spec),
        "segment_merge": {"production": {"last_step": 204000, "sha256": sha256_file(samples)}},
        "artifacts": [{**_record(stage / relative), "path": relative} for relative in
            ("stage_spec.json", "qc/state_point_qc.json", "samples/production.csv", "restart.final")]}
    _write(stage / "stage_manifest.json", manifest)
    receipt = {"schema_version": "polymer-smiles-attempt-receipt/v1", **task,
        "attempt_id": attempt.name, "attempt_root": str(attempt), "status": "FAILED",
        "ended_at": "2026-09-20T12:00:00+00:00", "stages": [
            {"stage": "prepare", "returncode": 0, "ended_at": "2026-09-20T11:00:00+00:00"},
            {"stage": "density", "returncode": 1, "ended_at": "2026-09-20T12:00:00+00:00"}],
        "artifacts": {key: _record(path) for key, path in (("request", attempt / "request.json"),
            ("prepared", prep / "prepared.json"), ("input_data", data), ("snapshot_metadata", metadata))}}
    _write(attempt / "receipt.json", receipt)
    selector = {"run_root": str(run_root), "run_id": run_id, "replica_id": "packing_001",
        "run_spec_sha256": sha256_file(run_root / "spec/run_spec.json"),
        "stage_manifest_sha256": sha256_file(stage / "stage_manifest.json"),
        "restart_sha256": sha256_file(stage / "restart.final"), "expected_step": 204000,
        "smiles_evidence": {key: _record(path) for key, path in (("request", attempt / "request.json"),
            ("receipt", attempt / "receipt.json"), ("prepared", prep / "prepared.json"))}}
    return {"task": task, "attempt": attempt, "selector": selector, "resolved": resolved,
            "execution_identity": identity, "stage": stage, "request": request, "prepared": prepared,
            "config": config, "run_root": run_root}


def _verify(fixture, selector=None, resolved=None, identity=None):
    with patch("subprocess.Popen", side_effect=AssertionError("parent validation must not start a model")):
        return verify_density_parent(selector or fixture["selector"], resolved or fixture["resolved"],
                                     identity or fixture["execution_identity"])


def test_catalog_parent_authenticated_and_frozen_50_ps_counted(tmp_path):
    f = make_smiles_parent(tmp_path)
    before = {str(p): p.read_bytes() for p in f["attempt"].rglob("*") if p.is_file()}
    result = _verify(f)
    assert result["smiles_identity"]["task_id"] == "S000001"
    assert result["smiles_identity"]["smiles"] == "*C*"
    assert result["cumulative_transition_ps"] == 50
    assert result["start_step"] == 204000
    assert result["parent_qc_status"] == "FAIL"
    assert result["old_qc_modified"] is False
    assert smiles_parent_selector(f["attempt"], f["task"])["expected_step"] == 204000
    assert before == {str(p): p.read_bytes() for p in f["attempt"].rglob("*") if p.is_file()}


@pytest.mark.parametrize("task_index", [2, 1691])
def test_catalog_membership_is_not_limited_to_first_smiles(tmp_path, task_index):
    f = make_smiles_parent(tmp_path, task_index=task_index)
    assert _verify(f)["smiles_identity"]["task_id"] == f"S{task_index:06d}"


@pytest.mark.parametrize("field", ["task_id", "smiles", "smiles_sha256"])
def test_catalog_exact_identity_rejects_invented_pair(tmp_path, field):
    f = make_smiles_parent(tmp_path)
    task = dict(f["task"])
    task[field] = "S001692" if field == "task_id" else "modified"
    with pytest.raises(ThermalConfigError, match="catalog|requested parent"):
        smiles_parent_selector(f["attempt"], task)


@pytest.mark.parametrize("key", ["request", "receipt", "prepared"])
def test_pinned_parent_documents_cannot_change(tmp_path, key):
    f = make_smiles_parent(tmp_path)
    path = Path(f["selector"]["smiles_evidence"][key]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ThermalConfigError, match="SMILES evidence"):
        _verify(f)


@pytest.mark.parametrize("change", ["active", "unended", "active_child", "wrong_attempt", "unbound_input"])
def test_repinned_terminal_receipt_still_must_authenticate_attempt(tmp_path, change):
    f = make_smiles_parent(tmp_path)
    path = f["attempt"] / "receipt.json"
    receipt = json.loads(path.read_text())
    if change == "active":
        receipt["status"] = "RUNNING"
    elif change == "unended":
        receipt.pop("ended_at")
    elif change == "active_child":
        receipt["stages"][-1]["returncode"] = None
    elif change == "wrong_attempt":
        receipt["attempt_id"] = "attempt_0002"
    else:
        receipt["artifacts"]["input_data"]["sha256"] = "0" * 64
    _write(path, receipt)
    f["selector"]["smiles_evidence"]["receipt"] = _record(path)
    with pytest.raises(ThermalConfigError):
        _verify(f)


@pytest.mark.parametrize("change", ["model", "snapshot", "metadata", "polymer", "runtime"])
def test_current_execution_must_match_original_parent(tmp_path, change):
    f = make_smiles_parent(tmp_path)
    current = copy.deepcopy(f["execution_identity"])
    resolved = copy.deepcopy(f["resolved"])
    if change == "model":
        current["model"]["sha256"] = "0" * 64
    elif change in {"snapshot", "metadata"}:
        current["input_snapshots"]["packing_001"][change + "_sha256"] = "0" * 64
    elif change == "polymer":
        resolved["system"]["polymer_id"] = "S000002"
    else:
        current["engine"]["runtime_dependency_files"][0]["sha256"] = "0" * 64
    with pytest.raises(ThermalConfigError, match="mismatch"):
        _verify(f, resolved=resolved, identity=current)


@pytest.mark.parametrize("target", ["restart.final", "samples/production.csv", "qc/state_point_qc.json"])
def test_parent_checkpoint_and_stage_artifacts_are_bound(tmp_path, target):
    f = make_smiles_parent(tmp_path)
    path = f["stage"] / target
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ThermalConfigError, match="mismatch"):
        _verify(f)


def test_smiles_parent_cannot_use_legacy_selector_bypass(tmp_path):
    f = make_smiles_parent(tmp_path)
    selector = dict(f["selector"])
    selector.pop("smiles_evidence")
    assert normalize_density_parent(selector) == selector
    with pytest.raises(ThermalConfigError, match="authenticated catalog SMILES"):
        _verify(f, selector=selector)


@pytest.mark.parametrize("relative", ["../outside", "replicas/packing_001/initialization", "replicas/other/mace_transition_npt"])
def test_selected_stage_path_cannot_escape_transition_whitelist(tmp_path, relative):
    f = make_smiles_parent(tmp_path)
    f["selector"]["stage_relative_path"] = relative
    with pytest.raises(ThermalConfigError, match="stage_relative_path"):
        _verify(f)


def test_original_snapshot_content_tamper_is_rejected(tmp_path):
    f = make_smiles_parent(tmp_path)
    Path(f["prepared"]["input_data"]).write_text("changed snapshot")
    with pytest.raises(ThermalConfigError, match="receipt artifact input_data"):
        _verify(f)


def test_legacy_polymer_selector_still_requires_no_smiles_keys(tmp_path):
    f = make_smiles_parent(tmp_path)
    selector = dict(f["selector"])
    selector.pop("smiles_evidence")
    assert set(normalize_density_parent(selector)) == {
        "run_root", "run_id", "replica_id", "run_spec_sha256", "stage_manifest_sha256",
        "restart_sha256", "expected_step"}


def _rewrite_parent(f, *, start_step=4000, production_steps=200000):
    """Re-sign synthetic artifacts after constructing an intentional fixture."""
    resolved, identity = f["resolved"], f["execution_identity"]
    resolved["execution_identity"] = identity
    resolved["resolved_spec_sha256"] = simulation.resolved_spec_sha256(resolved)
    _write(f["run_root"] / "spec/run_spec.json", resolved)
    stage = f["stage"]
    spec = json.loads((stage / "stage_spec.json").read_text())
    spec.update(execution_identity=identity, start_step=start_step, production_steps=production_steps)
    _write(stage / "stage_spec.json", spec)
    endpoint = start_step + production_steps
    samples = stage / "samples/production.csv"
    rows = samples.read_text().splitlines()
    rows[-1] = f"{endpoint},{endpoint * .00025},4,1000,1,10,10,10"
    samples.write_text("\n".join(rows) + "\n")
    manifest = json.loads((stage / "stage_manifest.json").read_text())
    manifest["stage_spec_sha256"] = canonical_sha256(spec)
    manifest["segment_merge"]["production"].update(last_step=endpoint, sha256=sha256_file(samples))
    manifest["artifacts"] = [{**_record(stage / record["path"]), "path": record["path"]}
                             for record in manifest["artifacts"]]
    _write(stage / "stage_manifest.json", manifest)
    f["selector"].update(run_spec_sha256=sha256_file(f["run_root"] / "spec/run_spec.json"),
        stage_manifest_sha256=sha256_file(stage / "stage_manifest.json"), expected_step=endpoint)


def test_previous_attempt_is_reauthenticated_and_50_plus_25_counted(tmp_path):
    original = make_smiles_parent(tmp_path)
    child = make_smiles_parent(tmp_path, attempt_number=2, parent=original)
    child["resolved"]["density_parent_restart"] = original["selector"]
    child["execution_identity"]["density_parent_restart"] = _verify(original)
    _rewrite_parent(child, start_step=204000, production_steps=100000)
    assert _verify(child)["cumulative_transition_ps"] == 75
    (original["stage"] / "restart.final").write_bytes(b"tampered ancestor")
    with pytest.raises(ThermalConfigError, match="SHA256 mismatch"):
        _verify(child)


def test_forged_parent_self_report_cannot_reset_budget(tmp_path):
    original = make_smiles_parent(tmp_path)
    child = make_smiles_parent(tmp_path, attempt_number=2, parent=original)
    child["resolved"]["density_parent_restart"] = original["selector"]
    child["execution_identity"]["density_parent_restart"] = _verify(original)
    child["execution_identity"]["density_parent_restart"]["cumulative_transition_ps"] = 0
    _rewrite_parent(child, start_step=204000, production_steps=100000)
    with pytest.raises(ThermalConfigError, match="authenticated previous parent intake identity"):
        _verify(child)


def test_continuation_helper_rejects_hard_qc_even_when_repinned(tmp_path):
    f = make_smiles_parent(tmp_path)
    qc_path = f["stage"] / "qc/state_point_qc.json"
    qc = json.loads(qc_path.read_text())
    checks = qc["policy_results"][0]["convergence"]["checks"]
    next(check for check in checks if check["name"] == "density_drift")["status"] = "FAIL"
    _write(qc_path, qc)
    _rewrite_parent(f)
    with pytest.raises(ThermalConfigError, match="sole failed check"):
        smiles_parent_selector(f["attempt"], f["task"])


def test_legacy_polymer_still_verifies_without_new_evidence(tmp_path):
    f = make_smiles_parent(tmp_path)
    f["selector"].pop("smiles_evidence")
    f["resolved"]["system"]["polymer_id"] = "P020001"
    _rewrite_parent(f)
    result = _verify(f)
    assert result["cumulative_transition_ps"] == 50
    assert "smiles_identity" not in result


def test_older_window_cannot_bypass_later_attempt_budget(tmp_path):
    f = make_smiles_parent(tmp_path)
    f["stage"].with_name("mace_transition_npt_sampling_0001").mkdir()
    with pytest.raises(ThermalConfigError, match="latest contiguous transition endpoint"):
        _verify(f)


def test_generated_sampling_history_retains_authenticated_ancestor_budget(tmp_path):
    from test_sampling_continuation import synthetic_process, qc_payload

    original = make_smiles_parent(tmp_path)
    task = original["task"]
    attempt = tmp_path / task["task_id"] / "attempt_0002"
    prep = attempt / "prep"
    prep.mkdir(parents=True)
    prepared = copy.deepcopy(original["prepared"])
    for field in ("input_data", "snapshot_metadata"):
        source = Path(prepared[field])
        destination = prep / source.name
        destination.write_bytes(source.read_bytes())
        prepared[field] = str(destination)
    request = {**original["request"], "attempt_id": attempt.name, "attempt_root": str(attempt),
        "prepared_manifest_path": str(prep / "prepared.json"),
        "density_parent_restart": original["selector"]}
    _write(attempt / "request.json", request)
    _write(prep / "prepared.json", prepared)
    config = copy.deepcopy(original["config"])
    config["system"].update({key: prepared[key] for key in ("input_data", "snapshot_metadata")})
    config["initialize"]["mace_transition_npt_ps"] = 25.
    config["density_parent_restart"] = original["selector"]
    config["sampling_continuation"] = {"increment_ps": 25., "max_transition_ps": 100., "max_density_ps": 100.}
    config["output_root"] = str(attempt / "density/native_runs")
    config_path = attempt / "density/mace_config.json"
    _write(config_path, config)
    run_id = smiles_native_run_id(request, attempt / "request.json")
    stages = []
    with patch("subprocess.Popen", side_effect=AssertionError("model execution forbidden")):
        with pytest.raises(simulation.SamplingNotConvergedError):
            simulation.execute_thermal_campaign(config_path, run_id=run_id,
                process_runner=synthetic_process(stages),
                qc_evaluator_factory=lambda spec: lambda _eq, _prod: qc_payload(spec, ("minimum_effective_samples",)))
    receipt = json.loads((original["attempt"] / "receipt.json").read_text())
    receipt.update(attempt_id=attempt.name, attempt_root=str(attempt), status="SAMPLING_BUDGET_EXHAUSTED")
    receipt["artifacts"] = {key: _record(path) for key, path in (
        ("request", attempt / "request.json"), ("prepared", prep / "prepared.json"),
        ("input_data", Path(prepared["input_data"])), ("snapshot_metadata", Path(prepared["snapshot_metadata"])))}
    _write(attempt / "receipt.json", receipt)
    run_root = Path(config["output_root"]) / (run_id + ".incomplete")
    stage = run_root / "replicas/packing_001/mace_transition_npt_sampling_0001"
    selector = {"run_id": run_id, "run_root": str(run_root), "replica_id": "packing_001",
        "run_spec_sha256": sha256_file(run_root / "spec/run_spec.json"),
        "stage_manifest_sha256": sha256_file(stage / "stage_manifest.json"),
        "restart_sha256": sha256_file(stage / "restart.final"), "expected_step": 404000,
        "stage_relative_path": stage.relative_to(run_root).as_posix(),
        "smiles_evidence": {key: _record(path) for key, path in (("request", attempt / "request.json"),
            ("receipt", attempt / "receipt.json"), ("prepared", prep / "prepared.json"))}}
    resolved = json.loads((run_root / "spec/run_spec.json").read_text())
    verified = verify_density_parent(selector, resolved, resolved["execution_identity"])
    assert verified["cumulative_transition_ps"] == 100
    assert verified["start_step"] == 404000
    assert verified["smiles_identity"]["snapshot_sha256"] == _verify(original)["smiles_identity"]["snapshot_sha256"]
    assert "sampling_window_0001" in verified["source_files"]
    assert len(stages) == 2
    with pytest.raises(ThermalConfigError, match="exhausted"):
        smiles_parent_selector(attempt, task)
