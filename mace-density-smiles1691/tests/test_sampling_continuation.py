"""Real filesystem and orchestration, synthetic physics only; no model is loaded."""
import csv
import json
from pathlib import Path
from unittest import mock

import pytest

from adapters import mace_density as adapter
from thermal_properties import simulation as sim
from thermal_properties.sampling_continuation import (
    REQUIRED_CHECKS, verify_sampling_history, verify_run_sampling_history,
)
from test_density_adapter import fixture


def synthetic_process(stages, *, interrupt_index=None, mutate=None):
    def run(invocation):
        stage = invocation.production_segment_path.parents[1]
        spec = json.loads((stage / "stage_spec.json").read_text())
        stages.append(spec)
        if interrupt_index is not None and len(stages) == interrupt_index:
            raise KeyboardInterrupt("controlled synthetic interruption")
        eq_end = spec["start_step"] + spec["equilibration_steps"]
        end = eq_end + spec["production_steps"]
        temperature = spec.get("state", {}).get("temperature_start_k", 305)
        for path, step in ((invocation.equilibration_segment_path, eq_end),
                           (invocation.production_segment_path, end)):
            row = {"step": step, "time_ps": step * .00025, "temp_K": temperature,
                   "press_bar": 1.01325, "density_g_cm3": 1., "volume_A3": 1000.,
                   "pe_eV": -100., "ke_eV": 10., "etotal_eV": -90., "enthalpy_eV": -89.,
                   "pxx_bar": 1.01325, "pyy_bar": 1.01325, "pzz_bar": 1.01325,
                   "pxy_bar": 0., "pxz_bar": 0., "pyz_bar": 0., "lx_A": 10.,
                   "ly_A": 10., "lz_A": 10., "fmax_eV_A": .1, "atom_count": 4}
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                sample_steps = (range(eq_end + spec.get("sample_every_steps", 100), end + 1,
                                      spec.get("sample_every_steps", 100))
                                if path == invocation.production_segment_path else [step])
                for sample_step in sample_steps:
                    writer.writerow({**row, "step": sample_step, "time_ps": sample_step * .00025})
        invocation.stdout_path.write_text("synthetic physics only")
        invocation.stderr_path.write_text("")
        invocation.log_path.write_text("synthetic physics only")
        if spec["equilibration_steps"] and "state" in spec:
            (stage / "restart.equilibration").write_bytes(f"equilibration {eq_end}".encode())
        invocation.final_restart_path.write_bytes(f"synthetic restart {end}".encode())
        if mutate:
            mutate(stage, spec)
        return 0
    return run


def qc_payload(spec, failed=()):
    checks = [{"name": name, "status": "FAIL" if name in failed else "PASS",
               "actual": 1, "criterion": 1} for name in sorted(REQUIRED_CHECKS)]
    status = "FAIL" if failed else "PASS"
    return {"status": status,
            "diagnostic_observables": {"all_finite": True, "missing_columns": []},
            "policy_results": [{**{key: policy[key] for key in ("role", "policy_id", "policy_sha256")},
                                "status": status, "convergence": {"status": status, "checks": checks}}
                               for policy in sim._statepoint_qc_policy_requirements(spec)]}


def prepare_config(fixture):
    root, request, prepared, site = fixture
    config = adapter.build_config(request, prepared, site, root / "density")
    config["sampling_continuation"] = {"increment_ps": 25., "max_transition_ps": 100., "max_density_ps": 100.}
    path = root / "sampling-config.json"
    path.write_text(json.dumps(config))
    return root, path


def run_campaign(config_path, stages, failures, *, interrupt_index=None):
    def evaluator(spec):
        role = spec["stage_role"]
        index = spec.get("sampling_window", {}).get("index", 0)
        failed = failures.get((role, index), ())
        return lambda _eq, _prod: qc_payload(spec, failed)
    with mock.patch("subprocess.Popen", side_effect=AssertionError("real model forbidden")):
        return sim.execute_thermal_campaign(config_path, run_id="synthetic",
            process_runner=synthetic_process(stages, interrupt_index=interrupt_index),
            qc_evaluator_factory=evaluator)


@pytest.mark.parametrize("transition_failures,density_failures,expected_windows", [
    (0, 0, (1, 1)), (1, 0, (2, 1)), (2, 0, (3, 1)), (0, 1, (1, 2)), (0, 3, (1, 4)),
])
def test_bounded_windows_preserve_evidence_and_select_only_latest_pass(
        fixture, transition_failures, density_failures, expected_windows):
    root, config = prepare_config(fixture)
    stages = []
    failures = {("mace_transition_npt", i): ("minimum_effective_samples",) for i in range(transition_failures)}
    failures.update({("npt_state_point", i): ("minimum_effective_samples",) for i in range(density_failures)})
    path = run_campaign(config, stages, failures)
    manifest = json.loads(path.read_text())
    assert manifest["qc_status"] == "PASS"
    history = verify_run_sampling_history(path.parents[1], manifest)
    transition = next(w for w in history.values() if w[0]["role"] == "transition")
    density = next(w for w in history.values() if w[0]["role"] == "density")
    assert (len(transition), len(density)) == expected_windows
    assert [w["duration_ps"] for w in transition] == [50.] + [25.] * transition_failures
    for group in (transition, density):
        assert all(w["qc_status"] == "FAIL" for w in group[:-1])
        assert group[-1]["qc_status"] == "PASS"
    assert density[-1]["analyzed_production_ps"] == 25. * expected_windows[1]
    production = path.parents[1] / density[-1]["qc_production"]["path"]
    with production.open() as handle:
        assert len(list(csv.DictReader(handle))) == 2500 * expected_windows[1]
    for spec in stages:
        if spec.get("sampling_window", {}).get("index", 0):
            assert spec["equilibration_steps"] == spec["ramp_steps"] == spec["constant_equilibration_steps"] == 0
            assert spec["initialize_velocities"] is False
            assert spec["production_steps"] == 100000
    assert len([s for s in stages if s["stage_role"] == "initialization"]) == 1
    record = next(iter(manifest["sampling_history"].values()))[0]
    old = path.parents[1] / record["path"]
    old.write_text(old.read_text() + " ")
    with pytest.raises(sim.StageIntegrityError, match="size mismatch|hash mismatch"):
        verify_run_sampling_history(path.parents[1], manifest)


@pytest.mark.parametrize("role,count", [("mace_transition_npt", 3), ("npt_state_point", 4)])
def test_exhaustion_preserves_complete_structured_outcome(fixture, role, count):
    root, config = prepare_config(fixture)
    stages = []
    with pytest.raises(sim.SamplingNotConvergedError) as error:
        run_campaign(config, stages, {(role, i): ("minimum_effective_samples",) for i in range(count)})
    assert error.value.code == "SAMPLING_BUDGET_EXHAUSTED"
    outcome = sim.verify_sampling_outcome(error.value.record_path)
    assert outcome["cumulative_ps"] == 100
    assert outcome["remaining_ps"] == 0
    assert len(outcome["history"]) == count
    assert len([s for s in stages if s["stage_role"] == role]) == count
    if role == "mace_transition_npt":
        assert not any(s["stage_role"] == "npt_state_point" for s in stages)
    checkpoint = error.value.record_path.parents[2] / outcome["checkpoint"]["path"]
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(sim.StageIntegrityError):
        sim.verify_sampling_outcome(error.value.record_path)


@pytest.mark.parametrize("failed", [("density_drift",), ("minimum_effective_samples", "mean_pressure_control")])
def test_hard_qc_never_continues(fixture, failed):
    _, config = prepare_config(fixture)
    stages = []
    with pytest.raises(sim.SamplingNotConvergedError) as error:
        run_campaign(config, stages, {("mace_transition_npt", 0): failed})
    assert error.value.code == "SAMPLING_QC_FAILED"
    assert len(stages) == 2
    assert sim.verify_sampling_outcome(error.value.record_path)["failed_checks"] == sorted(failed)


def test_missing_qc_evidence_does_not_enable_retry(fixture):
    _, config = prepare_config(fixture)
    stages = []
    with pytest.raises(sim.StageIntegrityError, match="diagnostic evidence"):
        sim.execute_thermal_campaign(config, run_id="synthetic",
            process_runner=synthetic_process(stages),
            qc_evaluator_factory=lambda spec: lambda _a, _b: "FAIL")
    assert len(stages) == 2


def test_controlled_interruption_stops_without_continuation_or_outcome(fixture):
    root, config = prepare_config(fixture)
    stages = []
    with pytest.raises(KeyboardInterrupt):
        run_campaign(config, stages, {("mace_transition_npt", 0): ("minimum_effective_samples",)}, interrupt_index=3)
    assert len(stages) == 3
    assert not list(root.rglob("outcome.json"))
    with pytest.raises(sim.ThermalSimulationError, match="cannot retry"):
        run_campaign(config, stages, {("mace_transition_npt", 0): ("minimum_effective_samples",)})
    assert len(stages) == 3


def test_config_policy_is_fixed_and_hash_bound(fixture):
    _, path = prepare_config(fixture)
    config = json.loads(path.read_text())
    resolved = sim.expand_thermal_config(config)
    digest = resolved["resolved_spec_sha256"]
    resolved["sampling_continuation"]["max_transition_ps"] = 125
    assert sim.resolved_spec_sha256(resolved) != digest
    for field, value in [("increment_ps", 50), ("max_transition_ps", 125), ("max_density_ps", True)]:
        changed = json.loads(path.read_text())
        changed["sampling_continuation"][field] = value
        with pytest.raises(sim.ThermalConfigError):
            sim.expand_thermal_config(changed)


def test_parent_50_ps_counts_against_budget_without_initialization(tmp_path):
    from test_smiles_parent_intake import make_smiles_parent

    parent = make_smiles_parent(tmp_path)
    config = dict(parent["config"])
    config["initialize"] = {**config["initialize"], "mace_transition_npt_ps": 25.}
    config["density_parent_restart"] = parent["selector"]
    config["sampling_continuation"] = {"increment_ps": 25., "max_transition_ps": 100., "max_density_ps": 100.}
    config["output_root"] = str(tmp_path / "child_runs")
    path = tmp_path / "child.json"
    path.write_text(json.dumps(config))
    stages = []
    before = {str(p): p.read_bytes() for p in parent["attempt"].rglob("*") if p.is_file()}
    with pytest.raises(sim.SamplingNotConvergedError) as error:
        run_campaign(path, stages, {("mace_transition_npt", i): ("minimum_effective_samples",) for i in range(2)})
    outcome = sim.verify_sampling_outcome(error.value.record_path)
    assert outcome["cumulative_ps"] == 100.
    assert len(stages) == 2
    assert all(s["stage_role"] == "mace_transition_npt" for s in stages)
    assert [s["start_step"] for s in stages] == [204000, 304000]
    assert all(s["production_steps"] == 100000 and s["equilibration_steps"] == 0 for s in stages)
    assert before == {str(p): p.read_bytes() for p in parent["attempt"].rglob("*") if p.is_file()}


def test_cumulative_density_qc_receives_exact_growing_population(fixture):
    _, config = prepare_config(fixture)
    stages, populations = [], []
    def evaluator(spec):
        def qc(_equil, production):
            if spec["stage_role"] == "npt_state_point":
                with production.open() as handle:
                    rows = list(csv.DictReader(handle))
                populations.append(len(rows))
                failed = ("minimum_effective_samples",) if len(rows) < 7500 else ()
            else:
                failed = ()
            return qc_payload(spec, failed)
        return qc
    manifest = sim.execute_thermal_campaign(config, run_id="synthetic",
        process_runner=synthetic_process(stages), qc_evaluator_factory=evaluator)
    assert populations == [2500, 5000, 7500]
    payload = json.loads(manifest.read_text())
    assert payload["state_points"][0]["analyzed_production_ps"] == 75.
    assert payload["state_points"][0]["production_steps"] == 100000


def test_hard_qc_at_density_extension_stops_without_third_window(fixture):
    _, config = prepare_config(fixture)
    stages = []
    with pytest.raises(sim.SamplingNotConvergedError) as error:
        run_campaign(config, stages, {
            ("npt_state_point", 0): ("minimum_effective_samples",),
            ("npt_state_point", 1): ("density_drift",),
        })
    outcome = sim.verify_sampling_outcome(error.value.record_path)
    assert outcome["code"] == "SAMPLING_QC_FAILED"
    assert outcome["cumulative_ps"] == 50.
    assert len([spec for spec in stages if spec["stage_role"] == "npt_state_point"]) == 2


def test_runtime_failure_is_never_retried(fixture):
    _, config = prepare_config(fixture)
    stages = []
    run = synthetic_process(stages)
    calls = []
    def runtime_failure(invocation):
        calls.append(invocation)
        return 17 if len(calls) == 2 else run(invocation)
    kwargs = dict(run_id="synthetic", process_runner=runtime_failure,
                  qc_evaluator_factory=lambda spec: lambda _a, _b: qc_payload(spec))
    with pytest.raises(sim.ThermalSimulationError):
        sim.execute_thermal_campaign(config, **kwargs)
    with pytest.raises(sim.ThermalSimulationError, match="cannot retry"):
        sim.execute_thermal_campaign(config, **kwargs)
    assert len(calls) == 2


@pytest.mark.parametrize("mutation", ["aggregate_value", "raw_stride", "raw_prefix", "raw_suffix"])
def test_cumulative_verifier_reconstructs_rows_even_when_modified_file_rehashed(fixture, mutation):
    from thermal_properties.sampling_continuation import _verify_cumulative_production

    _, config = prepare_config(fixture)
    manifest_path = run_campaign(config, [], {("npt_state_point", 0): ("minimum_effective_samples",)})
    root = manifest_path.parents[1]
    manifest = json.loads(manifest_path.read_text())
    histories = verify_run_sampling_history(root, manifest)
    windows = next(group for group in histories.values() if group[0]["role"] == "density")
    last = dict(windows[-1])
    spec_path = root / last["stage_spec"]["path"]
    spec = json.loads(spec_path.read_text())
    field = "qc_production" if mutation == "aggregate_value" else "production"
    path = root / last[field]["path"]
    with path.open() as handle:
        reader = csv.DictReader(handle)
        header, rows = reader.fieldnames, list(reader)
    if mutation == "aggregate_value":
        rows[0]["density_g_cm3"] = "1.234"
    elif mutation == "raw_stride":
        rows[1]["step"] = rows[0]["step"]
    elif mutation == "raw_prefix":
        rows = rows[1:]
    else:
        rows = rows[:-1]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    last[field] = sim.artifact_record(path, relative_to=root)
    with pytest.raises(sim.StageIntegrityError, match="exact verified source population|planned absolute cadence/count/bounds"):
        _verify_cumulative_production(root, last, windows[:-1], spec)
