"""Bounded prospective ESS-only sampling, with immutable per-window evidence.

Transition windows are assessed separately. Density QC uses the exact growing
post-equilibration production population, preserving each older failed result.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Mapping

from .provenance import canonical_sha256, sha256_file


WINDOW_SCHEMA = "thermal-properties-sampling-window/v1"
OUTCOME_SCHEMA = "thermal-properties-sampling-outcome/v1"
REQUIRED_CHECKS = frozenset({
    "structural_diagnostics_present", "minimum_production_samples",
    "all_required_values_finite", "positive_volume_and_density",
    "snapshot_contract_atom_count_present", "constant_positive_integer_atom_count",
    "atom_count_matches_snapshot_contract", "minimum_box_length", "box_aspect_ratio",
    "maximum_atomic_force", "physical_density_range", "strictly_increasing_steps",
    "strictly_increasing_times", "uniform_step_stride", "minimum_production_time_ps",
    "minimum_effective_samples", "minimum_complete_blocks", "mean_temperature_control",
    "mean_pressure_control", "density_drift", "block_mean_spread",
})


def _core():
    from . import simulation
    return simulation


def _immutable_json(path, payload):
    core = _core()
    if path.exists():
        if core._read_json(path) != payload:
            raise core.StageIntegrityError(f"sampling evidence changed: {path}")
    else:
        core._atomic_write_json(path, payload)


def _qc_failure_checks(stage_dir, spec):
    """Return authenticated failed checks; incomplete ESS claims fail closed."""
    core = _core()
    qc = core._read_json(stage_dir / "qc/state_point_qc.json")
    summary = core._policy_result_summary(qc)
    requirements = core._statepoint_qc_policy_requirements(spec)
    if ([{k: x[k] for k in ("role", "policy_id", "policy_sha256")} for x in summary]
            != [{k: x[k] for k in ("role", "policy_id", "policy_sha256")} for x in requirements]
            or qc.get("qc_implementation_sha256") != core.density_qc_implementation_sha256(spec["execution_identity"])):
        raise core.StageIntegrityError("sampling QC policy/implementation identity mismatch")
    if qc["status"] == "PASS":
        return []
    if qc.get("reason_codes"):
        raise core.StageIntegrityError("sampling QC contains unsupported evaluator/reason codes")
    diagnostics = qc.get("diagnostic_observables")
    if not isinstance(diagnostics, Mapping):
        raise core.StageIntegrityError("sampling QC lacks diagnostic evidence")
    failed = []
    if diagnostics.get("all_finite") is not True or diagnostics.get("missing_columns") != []:
        failed.append("diagnostic_observables")
    for policy in qc["policy_results"]:
        convergence = policy.get("convergence")
        if not isinstance(convergence, Mapping):
            raise core.StageIntegrityError("sampling QC lacks convergence evidence")
        checks = convergence.get("checks")
        if not isinstance(checks, list) or not checks:
            raise core.StageIntegrityError("sampling QC lacks individual checks")
        names = set()
        policy_failed = []
        for check in checks:
            if (not isinstance(check, Mapping) or not isinstance(check.get("name"), str)
                    or check["name"] in names or check.get("status") not in {"PASS", "FAIL"}
                    or "actual" not in check or "criterion" not in check):
                raise core.StageIntegrityError("sampling QC check is incomplete or duplicated")
            names.add(check["name"])
            if check["status"] == "FAIL":
                policy_failed.append(check["name"])
        expected_status = "FAIL" if policy_failed else "PASS"
        if convergence.get("status") != expected_status:
            raise core.StageIntegrityError("sampling convergence/check status mismatch")
        if policy.get("status") != expected_status and not failed:
            raise core.StageIntegrityError("sampling policy/check status mismatch")
        if set(policy_failed) == {"minimum_effective_samples"} and not REQUIRED_CHECKS.issubset(names):
            raise core.StageIntegrityError("ESS-only QC requires every structural and stationarity check")
        failed.extend(policy_failed)
    if not failed:
        raise core.StageIntegrityError("failed sampling QC has no supported failed checks")
    return sorted(set(failed))


def _verify_raw_production_cadence(root, window):
    core = _core()
    spec = core._read_json(core.verify_artifact_record(window["stage_spec"], relative_to=root))
    source = core.verify_artifact_record(window["production"], relative_to=root)
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    stride = int(spec["sample_every_steps"])
    start = int(spec["start_step"]) + int(spec["equilibration_steps"])
    end = start + int(spec["production_steps"])
    # LAMMPS fix print writes at absolute multiples of its cadence. The
    # production merge removes the phase-start row with an exclusive cutoff;
    # the final production timestep is retained when it lies on the cadence.
    expected_steps = list(range((start // stride + 1) * stride, end + 1, stride))
    if [int(row["step"]) for row in rows] != expected_steps:
        raise core.StageIntegrityError("cumulative source samples violate planned absolute cadence/count/bounds")
    return rows


def _verify_cumulative_production(root, window, previous_windows, spec):
    """Reconstruct the exact row population without trusting a merge summary."""
    core = _core()
    raw_path = core.verify_artifact_record(window["production"], relative_to=root)
    qc_path = core.verify_artifact_record(window["qc_production"], relative_to=root)
    expected_sources = [item["production"] for item in previous_windows]
    if window["role"] == "transition" or not previous_windows:
        if raw_path != qc_path or spec.get("sampling_production_sources") is not None:
            raise core.StageIntegrityError("unexpected cumulative production population")
        return
    if spec.get("sampling_production_sources") != expected_sources:
        raise core.StageIntegrityError("cumulative production omits or changes a historical window")
    if (raw_path.parent.parent / spec.get("sampling_run_root_relative", "")).resolve() != Path(root).resolve():
        raise core.StageIntegrityError("cumulative production root mismatch")
    if qc_path != raw_path.with_name("production.cumulative.csv"):
        raise core.StageIntegrityError("cumulative production path is not canonical")
    for source_window in [*previous_windows, window]:
        _verify_raw_production_cadence(root, source_window)
    expected_rows = []
    fieldnames = None
    for record in [*expected_sources, window["production"]]:
        source = core.verify_artifact_record(record, relative_to=root)
        with source.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            if reader.fieldnames != fieldnames:
                raise core.StageIntegrityError("cumulative production columns changed")
            expected_rows.extend(reader)
    steps = [int(row["step"]) for row in expected_rows]
    stride = int(spec["sample_every_steps"])
    if not steps or any(right - left != stride for left, right in zip(steps, steps[1:])):
        raise core.StageIntegrityError("cumulative production has duplicate, missing, or nonuniform steps")
    with qc_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fieldnames or list(reader) != expected_rows:
            raise core.StageIntegrityError("cumulative production is not the exact verified source population")


def verify_sampling_history(root, records, expected_stage_id=None):
    """Verify every window, stage artifact, checkpoint chain and cumulative cap."""
    core = _core()
    root = Path(root).resolve()
    if not isinstance(records, list) or not records:
        raise core.StageIntegrityError("sampling history is empty")
    windows = []
    for index, record in enumerate(records):
        path = core.verify_artifact_record(record, relative_to=root)
        window = core._read_json(path)
        if window.get("schema_version") != WINDOW_SCHEMA or window.get("index") != index:
            raise core.StageIntegrityError("sampling history index/schema mismatch")
        if expected_stage_id is not None and window.get("stage_id") != expected_stage_id:
            raise core.StageIntegrityError("sampling history stage identity mismatch")
        expected_path = root / "sampling_history" / window["stage_id"] / f"window_{index:04d}.json"
        if path != expected_path.resolve():
            raise core.StageIntegrityError("sampling history record has noncanonical path")
        run_spec_path = core.verify_artifact_record(window["run_spec"], relative_to=root)
        resolved = core._read_json(run_spec_path)
        if resolved.get("resolved_spec_sha256") != core.resolved_spec_sha256(resolved):
            raise core.StageIntegrityError("sampling resolved spec digest mismatch")
        policy = resolved.get("sampling_continuation")
        if policy != {"increment_ps": 25.0, "max_transition_ps": 100.0, "max_density_ps": 100.0}:
            raise core.StageIntegrityError("sampling history lacks the fixed authorized budget")
        manifest_path = core.verify_artifact_record(window["stage_manifest"], relative_to=root)
        spec_path = core.verify_artifact_record(window["stage_spec"], relative_to=root)
        checkpoint = core.verify_artifact_record(window["checkpoint"], relative_to=root)
        qc_path = core.verify_artifact_record(window["qc"], relative_to=root)
        input_checkpoint = core.verify_artifact_record(window["input_checkpoint"], relative_to=root)
        stage = manifest_path.parent
        spec = core._read_json(spec_path)
        manifest = core._read_json(manifest_path)
        if (spec_path != stage / "stage_spec.json" or checkpoint != stage / "restart.final"
                or qc_path != stage / "qc/state_point_qc.json"
                or spec.get("stage_id") != window["stage_id"]
                or spec.get("execution_identity") != resolved.get("execution_identity")
                or not core._stage_is_reusable(stage, canonical_sha256(spec))
                or manifest.get("execution_status") != "COMPLETE"
                or manifest.get("qc_status") != window.get("qc_status")):
            raise core.StageIntegrityError("sampling stage identity/artifacts mismatch")
        role = "transition" if spec.get("stage_role") == "mace_transition_npt" else "density"
        parent = resolved.get("execution_identity", {}).get("density_parent_restart")
        if parent and index == 0:
            verified_parent = core.verify_density_parent(resolved["density_parent_restart"], resolved,
                                                         resolved["execution_identity"])
            if verified_parent != parent:
                raise core.StageIntegrityError("sampling history parent identity/budget changed")
        initial = float(parent["cumulative_transition_ps"]) if role == "transition" and parent else 0.0
        duration = float(spec["production_steps"]) * float(resolved["engine"]["dt_ps"])
        cumulative = initial + sum(x["duration_ps"] for x in windows) + duration
        expected_duration = 50.0 if role == "transition" and initial == 0 and index == 0 else 25.0
        previous = records[index - 1] if index else None
        if (duration != expected_duration or window.get("duration_ps") != duration
                or window.get("initial_cumulative_ps") != initial
                or window.get("cumulative_ps") != cumulative or cumulative > 100
                or window.get("max_ps") != 100 or window.get("remaining_ps") != 100 - cumulative
                or window.get("role") != role
                or window.get("previous_window_record") != previous
                or spec.get("sampling_window") != {"index": index, "role": role,
                    "initial_cumulative_ps": initial, "max_ps": 100.0, "increment_ps": 25.0,
                    "previous_window_record": previous}
                or input_checkpoint != core._resolve_stage_reference(stage, spec["predecessor_restart"])
                or spec.get("predecessor_restart_sha256") != sha256_file(input_checkpoint)):
            raise core.StageIntegrityError("sampling budget/checkpoint lineage mismatch")
        if index:
            previous_window = windows[-1]
            if (previous_window["stage_id"] != window["stage_id"]
                    or previous_window["failed_checks"] != ["minimum_effective_samples"]
                    or previous_window["checkpoint"] != window["input_checkpoint"]
                    or previous_window["end_step"] != int(spec["start_step"])
                    or any(spec.get(k) != 0 for k in ("ramp_steps", "constant_equilibration_steps", "equilibration_steps"))
                    or spec.get("initialize_velocities") is not False):
                raise core.StageIntegrityError("sampling continuation changed dynamics or continued hard QC")
        end_step = int(spec["start_step"]) + int(spec["equilibration_steps"]) + int(spec["production_steps"])
        failed = _qc_failure_checks(stage, spec)
        if window.get("failed_checks") != failed or window.get("end_step") != end_step:
            raise core.StageIntegrityError("sampling QC/end-step evidence mismatch")
        production = core.verify_artifact_record(window["production"], relative_to=root)
        if production != stage / "samples/production.csv":
            raise core.StageIntegrityError("sampling raw production is not canonical")
        _verify_cumulative_production(root, window, windows, spec)
        analyzed = cumulative if role == "density" else duration
        if window.get("analyzed_production_ps") != analyzed:
            raise core.StageIntegrityError("sampling analyzed duration mismatch")
        windows.append(window)
    return windows


def verify_sampling_outcome(path):
    core = _core()
    path = Path(path).resolve()
    payload = core._read_json(path)
    if payload.get("schema_version") != OUTCOME_SCHEMA:
        raise core.StageIntegrityError("unrecognized sampling outcome schema")
    root = (path.parent / payload["run_root_relative"]).resolve()
    if path != root / "sampling_history" / payload["stage_id"] / "outcome.json":
        raise core.StageIntegrityError("sampling outcome path is not canonical")
    windows = verify_sampling_history(root, payload["history"], payload["stage_id"])
    last = windows[-1]
    expected_code = ("SAMPLING_BUDGET_EXHAUSTED" if last["cumulative_ps"] == last["max_ps"]
                     else "NEEDS_MORE_SAMPLING") if last["failed_checks"] == ["minimum_effective_samples"] else "SAMPLING_QC_FAILED"
    if (not last["failed_checks"] or payload.get("code") != expected_code
            or payload.get("status") != expected_code or payload.get("execution_status") != "COMPLETE"
            or payload.get("qc_status") != "FAIL"
            or any(payload.get(key) != last.get(key) for key in
                   ("stage_role", "cumulative_ps", "max_ps", "remaining_ps", "failed_checks", "checkpoint"))):
        raise core.StageIntegrityError("sampling outcome is inconsistent with its completed window")
    return payload


def verify_run_sampling_history(root, manifest):
    """Bind complete historical coverage to the selected final stage records."""
    core = _core()
    root = Path(root)
    if "resolved_run_spec" not in manifest:
        if (root / "spec/run_spec.json").exists() or "sampling_history" in manifest or "sampling_continuation" in manifest:
            raise core.StageIntegrityError("sampling resolved spec binding is missing")
        return {}
    resolved = core._read_json(core.verify_artifact_record(manifest["resolved_run_spec"], relative_to=root))
    enabled = resolved.get("sampling_continuation")
    if enabled is None:
        if "sampling_history" in manifest or "sampling_continuation" in manifest:
            raise core.StageIntegrityError("unexpected sampling continuation in manifest")
        return {}
    if manifest.get("sampling_continuation") != enabled:
        raise core.StageIntegrityError("sampling policy is missing from final manifest")
    expected = {item["stage_id"] for item in resolved["plan"]["initialization"]
                if item["stage_role"] == "mace_transition_npt"}
    expected.update(f'{item["replica_id"]}__{item["state"]["state_id"]}' for item in resolved["plan"]["steps"])
    histories = manifest.get("sampling_history")
    if not isinstance(histories, Mapping) or set(histories) != expected:
        raise core.StageIntegrityError("sampling history does not cover every planned stage")
    windows_by_stage = {}
    for stage_id, records in histories.items():
        windows = verify_sampling_history(root, records, stage_id)
        last = windows[-1]
        if last["qc_status"] != "PASS" or last["failed_checks"]:
            raise core.StageIntegrityError("selected sampling window did not pass QC")
        selected = [item for item in manifest.get("initialization_stages", []) if item.get("stage_id") == stage_id]
        if selected:
            if len(selected) != 1 or selected[0].get("stage_manifest") != last["stage_manifest"]:
                raise core.StageIntegrityError("selected transition does not match sampling history")
        else:
            selected = [item for item in manifest.get("state_points", []) if item.get("state_point_id") == stage_id]
            if (len(selected) != 1
                    or selected[0].get("artifacts", {}).get("state_point_qc", {}).get("sha256") != last["qc"]["sha256"]
                    or any(selected[0].get("artifacts", {}).get("thermo_samples", {}).get(key) != last["qc_production"][key]
                           for key in ("path", "sha256", "bytes"))
                    or selected[0].get("analyzed_production_ps") != last["analyzed_production_ps"]):
                raise core.StageIntegrityError("selected density stage does not match sampling history")
        windows_by_stage[stage_id] = windows
    return windows_by_stage


def execute_sampling_stage(*, resolved, run_root, stage_dir, stage_spec, command,
                           environment, process_runner, evaluator_factory, input_root):
    """Execute bounded windows; never catch runtime failures or interruptions."""
    core = _core()
    root, original_dir = Path(run_root), Path(stage_dir)
    role = "transition" if stage_spec["stage_role"] == "mace_transition_npt" else "density"
    parent = resolved.get("execution_identity", {}).get("density_parent_restart")
    initial = float(parent["cumulative_transition_ps"]) if role == "transition" and parent else 0.0
    history = []
    spec = dict(stage_spec)
    cumulative = initial
    for index in range(4):
        stage = original_dir if index == 0 else original_dir.with_name(f"{original_dir.name}_sampling_{index:04d}")
        duration = int(spec["production_steps"]) * float(resolved["engine"]["dt_ps"])
        if cumulative + duration > 100:
            raise core.StageIntegrityError("sampling request would exceed authenticated cumulative budget")
        spec["sampling_window"] = {"index": index, "role": role,
            "initial_cumulative_ps": initial, "max_ps": 100.0, "increment_ps": 25.0,
            "previous_window_record": history[-1] if history else None}
        if role == "density" and history:
            previous_windows = verify_sampling_history(root, history, spec["stage_id"])
            for previous_window in previous_windows:
                _verify_raw_production_cadence(root, previous_window)
            spec["sampling_production_sources"] = [window["production"] for window in previous_windows]
            spec["sampling_run_root_relative"] = os.path.relpath(root, stage)
        predecessor = core._resolve_stage_reference(stage, spec["predecessor_restart"])
        spec["predecessor_restart_sha256"] = sha256_file(predecessor)
        if history:
            verify_sampling_history(root, history, spec["stage_id"])
        core._verify_execution_identity(resolved, input_root)
        status_path = stage / "stage_status.json"
        if status_path.exists() and core._read_json(status_path).get("execution_status") != "COMPLETE":
            raise core.ThermalSimulationError("bounded sampling cannot retry an interrupted/runtime-failed window")
        stage_environment = dict(environment)
        for key, filename in (("THERMAL_EQUIL_RESTART_ROOT", "checkpoints/equilibration.checkpoint.*.restart"),
                              ("THERMAL_PROD_RESTART_ROOT", "checkpoints/production.checkpoint.*.restart"),
                              ("THERMAL_EQUIL_FINAL_RESTART", "restart.equilibration")):
            stage_environment[key] = str((stage / filename).resolve())
        result = core.execute_npt_stage(stage_dir=stage, stage_spec=spec, command=command,
            environment=stage_environment, process_runner=process_runner,
            qc_evaluator=evaluator_factory(spec))
        cumulative += duration
        persisted_spec = core._read_json(stage / "stage_spec.json")
        if not core._stage_is_reusable(stage, canonical_sha256(persisted_spec)):
            raise core.StageIntegrityError("sampling window did not complete")
        failed = _qc_failure_checks(stage, persisted_spec)
        artifact = lambda p: core.artifact_record(p, relative_to=root)
        window = {"schema_version": WINDOW_SCHEMA, "index": index, "stage_id": spec["stage_id"],
            "stage_role": spec["stage_role"], "role": role, "duration_ps": duration,
            "analyzed_production_ps": cumulative if role == "density" else duration,
            "initial_cumulative_ps": initial, "cumulative_ps": cumulative, "max_ps": 100.0,
            "remaining_ps": 100.0 - cumulative, "qc_status": result.qc_status, "failed_checks": failed,
            "end_step": int(spec["start_step"]) + int(spec["equilibration_steps"]) + int(spec["production_steps"]),
            "previous_window_record": history[-1] if history else None,
            "run_spec": artifact(root / "spec/run_spec.json"),
            "stage_manifest": artifact(stage / "stage_manifest.json"),
            "stage_spec": artifact(stage / "stage_spec.json"), "qc": artifact(stage / "qc/state_point_qc.json"),
            "checkpoint": artifact(stage / "restart.final"), "input_checkpoint": artifact(predecessor),
            "production": artifact(stage / "samples/production.csv"),
            "qc_production": artifact(stage / "samples" / ("production.cumulative.csv"
                if spec.get("sampling_production_sources") else "production.csv"))}
        window_path = root / "sampling_history" / spec["stage_id"] / f"window_{index:04d}.json"
        _immutable_json(window_path, window)
        history.append(artifact(window_path))
        verify_sampling_history(root, history, spec["stage_id"])
        if not failed:
            return result, spec, history
        if failed != ["minimum_effective_samples"] or cumulative == 100:
            code = "SAMPLING_BUDGET_EXHAUSTED" if failed == ["minimum_effective_samples"] else "SAMPLING_QC_FAILED"
            outcome = {"schema_version": OUTCOME_SCHEMA, "status": code, "code": code,
                "run_root_relative": "../..", "execution_status": "COMPLETE", "qc_status": "FAIL",
                **{key: window[key] for key in ("stage_id", "stage_role", "cumulative_ps", "max_ps", "remaining_ps", "failed_checks", "checkpoint")},
                "history": history}
            outcome_path = root / "sampling_history" / spec["stage_id"] / "outcome.json"
            _immutable_json(outcome_path, outcome)
            verify_sampling_outcome(outcome_path)
            raise core.SamplingNotConvergedError(code, outcome_path)
        next_stage = original_dir.with_name(f"{original_dir.name}_sampling_{index + 1:04d}")
        spec = dict(spec)
        spec.update(start_step=window["end_step"], ramp_steps=0, constant_equilibration_steps=0,
            equilibration_steps=0, production_steps=core.ps_to_steps(25, resolved["engine"]["dt_ps"]),
            initialize_velocities=False, predecessor_restart=os.path.relpath(stage / "restart.final", next_stage),
            working_directory=os.path.relpath(input_root, next_stage),
            temperature_ramp_start_k=spec["temperature_ramp_end_k"])
        spec["state"] = {**spec["state"], "duration_ps": "25"}
    raise core.StageIntegrityError("sampling loop exceeded the fixed budget")
