"""Integrity-checked, read-only summaries of finalized thermal runs."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

from .provenance import sha256_file
from .simulation import ThermalSimulationError, verify_artifact_record


RUN_MANIFEST_SCHEMA = "thermal-properties-run-manifest/v1"
RUN_STATUS_SCHEMA = "thermal-properties-run-status/v1"
STATE_POINT_QC_SCHEMA = "thermal-properties-state-point-qc/v1"


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ThermalSimulationError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ThermalSimulationError(f"{label} root must be an object: {path}")
    return value


def _verify_manifest(path: Path, manifest: Mapping[str, Any]) -> str:
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA:
        raise ThermalSimulationError(
            f"unsupported run manifest schema: {manifest.get('schema_version')!r}"
        )
    actual = sha256_file(path)
    sidecar = path.with_name("run_manifest.sha256")
    try:
        declared = sidecar.read_text(encoding="ascii").strip().lower()
    except (OSError, UnicodeError) as exc:
        raise ThermalSimulationError(
            f"cannot read run-manifest digest: {sidecar}: {exc}"
        ) from exc
    if declared != actual:
        raise ThermalSimulationError(f"run-manifest digest mismatch: {path}")
    return actual


def _policy_summary(policy: Mapping[str, Any]) -> dict[str, Any]:
    convergence = policy.get("convergence")
    convergence = convergence if isinstance(convergence, Mapping) else {}
    density = convergence.get("density")
    density = density if isinstance(density, Mapping) else {}
    checks = convergence.get("checks")
    checks = checks if isinstance(checks, list) else []
    failed_checks = [
        str(check.get("name"))
        for check in checks
        if isinstance(check, Mapping) and str(check.get("status", "")).upper() != "PASS"
    ]
    return {
        "role": policy.get("role"),
        "policy_id": policy.get("policy_id"),
        "status": str(policy.get("status", "NOT_EVALUATED")).upper(),
        "sample_count": convergence.get("sample_count"),
        "production_time_ps": convergence.get("production_time_ps"),
        "target_temperature_K": convergence.get("target_temperature_K"),
        "observed_mean_temperature_K": convergence.get(
            "observed_mean_temperature_K"
        ),
        "target_pressure_bar": convergence.get("target_pressure_bar"),
        "observed_mean_pressure_bar": convergence.get("observed_mean_pressure_bar"),
        "density_mean_g_cm3": density.get("mean"),
        "density_standard_error_g_cm3": density.get("standard_error"),
        "density_effective_sample_count": density.get("effective_sample_count"),
        "density_drift_g_cm3_per_ps": convergence.get(
            "density_drift_g_cm3_per_ps"
        ),
        "failed_checks": failed_checks,
    }


def _state_summary(run_root: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = state.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ThermalSimulationError("state point has no artifacts object")
    qc_record = artifacts.get("state_point_qc")
    if not isinstance(qc_record, Mapping):
        raise ThermalSimulationError("state point has no state_point_qc artifact")
    qc_path = verify_artifact_record(qc_record, relative_to=run_root)
    qc = _read_object(qc_path, "state-point QC")
    if qc.get("schema_version") != STATE_POINT_QC_SCHEMA:
        raise ThermalSimulationError(
            f"unsupported state-point QC schema: {qc.get('schema_version')!r}"
        )
    qc_status = str(qc.get("status", "NOT_EVALUATED")).upper()
    manifest_qc_status = str(state.get("qc_status", "NOT_EVALUATED")).upper()
    if qc_status != manifest_qc_status:
        raise ThermalSimulationError(
            f"state-point QC status mismatch: {state.get('state_point_id')}"
        )
    raw_policies = qc.get("policy_results")
    if not isinstance(raw_policies, list) or not raw_policies:
        raise ThermalSimulationError(
            f"state-point QC has no policy results: {state.get('state_point_id')}"
        )
    policies = [
        _policy_summary(item)
        for item in raw_policies
        if isinstance(item, Mapping)
    ]
    if len(policies) != len(raw_policies):
        raise ThermalSimulationError(
            f"state-point QC policy result is malformed: {state.get('state_point_id')}"
        )
    diagnostics = qc.get("diagnostic_observables")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    return {
        "state_point_id": state.get("state_point_id"),
        "replica_id": state.get("replica_id"),
        "roles": state.get("roles"),
        "target_temperature_K": state.get("target_temperature_K"),
        "target_pressure_bar": state.get("target_pressure_bar"),
        "execution_status": state.get("status"),
        "qc_status": qc_status,
        "maximum_force_eV_A": diagnostics.get("maximum_force_eV_A"),
        "qc_artifact": {
            "path": str(qc_path),
            "sha256": qc_record.get("sha256"),
            "integrity": "VERIFIED",
        },
        "policy_results": policies,
    }


def build_run_status(run_manifest: str | Path) -> dict[str, Any]:
    """Return a deterministic status report without launching analysis or MD."""

    manifest_path = Path(run_manifest).resolve()
    manifest = _read_object(manifest_path, "run manifest")
    manifest_sha256 = _verify_manifest(manifest_path, manifest)
    run_root = manifest_path.parent.parent.resolve()
    # New bounded runs must authenticate every prior window, not only the
    # selected final PASS. Keep legacy summary-only fixtures compatible.
    if ("resolved_run_spec" in manifest or "sampling_history" in manifest
            or "sampling_continuation" in manifest
            or (run_root / "spec/run_spec.json").exists()):
        from .sampling_continuation import verify_run_sampling_history
        try:
            verify_run_sampling_history(run_root, manifest)
        except (KeyError, TypeError, ValueError) as exc:
            raise ThermalSimulationError(f"invalid bounded sampling evidence: {exc}") from exc

    raw_states = manifest.get("state_points")
    if not isinstance(raw_states, list):
        raise ThermalSimulationError("run manifest state_points must be an array")
    states = [
        _state_summary(run_root, state)
        for state in raw_states
        if isinstance(state, Mapping)
    ]
    if len(states) != len(raw_states):
        raise ThermalSimulationError("run manifest contains a malformed state point")

    raw_initializers = manifest.get("initialization_stages", [])
    if not isinstance(raw_initializers, list):
        raise ThermalSimulationError(
            "run manifest initialization_stages must be an array"
        )
    initializers = []
    for item in raw_initializers:
        if not isinstance(item, Mapping):
            raise ThermalSimulationError(
                "run manifest contains a malformed initialization stage"
            )
        gate = item.get("qc_gate")
        initializers.append(
            {
                "stage_id": item.get("stage_id"),
                "stage_role": item.get("stage_role"),
                "replica_id": item.get("replica_id"),
                "qc_status": item.get("qc_status"),
                "qc_gate_decision": (
                    gate.get("decision") if isinstance(gate, Mapping) else None
                ),
                "property_branch_started": (
                    gate.get("property_branch_started")
                    if isinstance(gate, Mapping)
                    else None
                ),
            }
        )

    gate_reasons = []
    if manifest.get("finalized") is not True:
        gate_reasons.append("RUN_NOT_FINALIZED")
    if str(manifest.get("execution_status", "")).upper() != "COMPLETE":
        gate_reasons.append("EXECUTION_NOT_COMPLETE")
    if str(manifest.get("qc_status", "")).upper() != "PASS":
        gate_reasons.append("RUN_QC_NOT_PASS")
    if manifest.get("run_class") != "PRODUCTION":
        gate_reasons.append("RUN_CLASS_NOT_PRODUCTION")
    if manifest.get("scientific_eligible") is not True:
        gate_reasons.append("SCIENTIFIC_ELIGIBLE_NOT_TRUE")

    state_counts = Counter(item["qc_status"] for item in states)
    initializer_counts = Counter(
        str(item.get("qc_status", "NOT_EVALUATED")).upper()
        for item in initializers
    )
    return {
        "schema_version": RUN_STATUS_SCHEMA,
        "run_manifest": str(manifest_path),
        "run_manifest_sha256": manifest_sha256,
        "manifest_integrity": "VERIFIED",
        "run_id": manifest.get("run_id"),
        "thermal_mode": manifest.get("thermal_mode"),
        "status": manifest.get("status"),
        "execution_status": manifest.get("execution_status"),
        "qc_status": manifest.get("qc_status"),
        "analysis_status": manifest.get("analysis_status"),
        "run_class": manifest.get("run_class"),
        "scientific_eligible": manifest.get("scientific_eligible"),
        "scientific_gate_passed": not gate_reasons,
        "scientific_gate_blocking_reasons": gate_reasons,
        "initialization_qc_counts": dict(sorted(initializer_counts.items())),
        "state_point_qc_counts": dict(sorted(state_counts.items())),
        "initialization_stages": initializers,
        "state_points": states,
    }
