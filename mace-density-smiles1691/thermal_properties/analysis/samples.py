"""Read-only loading, integrity checking, and output provenance helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

AMU_PER_ANGSTROM3_TO_G_CM3 = 1.66053906660
REQUIRED_THERMO_COLUMNS = (
    "phase",
    "step",
    "time_ps",
    "temp_K",
    "press_bar",
    "volume_A3",
)


class AnalysisInputError(RuntimeError):
    """Base class for deterministic input or provenance failures."""

    reason_code = "ANALYSIS_INPUT_ERROR"

    def __init__(self, message: str, *, reason_code: str | None = None) -> None:
        super().__init__(message)
        if reason_code is not None:
            self.reason_code = reason_code


class ArtifactIntegrityError(AnalysisInputError):
    """A source artifact is absent, escapes its run, or fails its digest."""

    reason_code = "HASH_MISMATCH"


class ManifestSchemaError(AnalysisInputError):
    """The run manifest does not provide the required analysis contract."""

    reason_code = "SCHEMA_UNSUPPORTED"


@dataclass(frozen=True)
class LoadedSamples:
    """A state point's samples together with immutable source provenance."""

    state_point_id: str
    frame: pd.DataFrame
    artifact: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestSchemaError(
            f"cannot read JSON {source}: {exc}", reason_code="MANIFEST_MISSING"
        ) from exc
    if not isinstance(value, dict):
        raise ManifestSchemaError(f"JSON root must be an object: {source}")
    return value


def verify_run_manifest_digest(
    manifest_path: str | Path, manifest: Mapping[str, Any]
) -> str:
    """Verify the detached digest required by the canonical run schema."""

    source = Path(manifest_path).resolve()
    actual = sha256_file(source)
    if manifest.get("schema_version") != "thermal-properties-run-manifest/v1":
        return actual
    sidecar = source.with_name("run_manifest.sha256")
    if not sidecar.is_file():
        raise ArtifactIntegrityError(
            f"canonical run manifest lacks detached digest: {sidecar}",
            reason_code="MANIFEST_DIGEST_MISSING",
        )
    try:
        declared = sidecar.read_text(encoding="ascii").strip().lower()
    except (OSError, UnicodeError) as exc:
        raise ArtifactIntegrityError(
            f"cannot read run-manifest digest: {sidecar}",
            reason_code="HASH_MISMATCH",
        ) from exc
    if (
        len(declared) != 64
        or any(character not in "0123456789abcdef" for character in declared)
        or declared != actual
    ):
        raise ArtifactIntegrityError(
            f"run-manifest digest mismatch: {source}",
            reason_code="HASH_MISMATCH",
        )
    return actual


def require_scientifically_eligible_run(manifest: Mapping[str, Any]) -> None:
    """Fail closed before property analysis for smoke/debug/pilot data."""

    run_class = manifest.get("run_class")
    scientific_eligible = manifest.get("scientific_eligible")
    reuse_eligible = manifest.get("reuse_eligible")
    if (
        run_class not in {"ENGINEERING_SMOKE", "DEBUG", "PILOT", "PRODUCTION"}
        or type(scientific_eligible) is not bool
        or type(reuse_eligible) is not bool
    ):
        raise ManifestSchemaError(
            "run manifest lacks explicit run classification and eligibility gates",
            reason_code="RUN_CLASSIFICATION_MISSING",
        )
    if run_class != "PRODUCTION" or scientific_eligible is not True:
        raise AnalysisInputError(
            f"{run_class} run is not eligible for scientific property analysis",
            reason_code="RUN_NOT_SCIENTIFICALLY_ELIGIBLE",
        )
    if reuse_eligible and not scientific_eligible:
        raise ManifestSchemaError(
            "reuse_eligible=true requires scientific_eligible=true",
            reason_code="RUN_CLASSIFICATION_INVALID",
        )


def json_safe(value: Any) -> Any:
    """Convert numpy/pandas/path values into strict JSON-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def write_json_new(path: str | Path, value: Any) -> None:
    """Create a JSON file and refuse to replace anything already present."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(
            json_safe(value),
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        handle.write("\n")


def write_csv_new(path: str | Path, frame: pd.DataFrame) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)


def create_analysis_output_dir(
    output_root: str | Path, analysis_type: str
) -> tuple[str, Path]:
    """Create an unguessable output directory without an overwrite path."""

    parent = Path(output_root).resolve() / analysis_type
    parent.mkdir(parents=True, exist_ok=True)
    for _ in range(16):
        analysis_id = str(uuid.uuid4())
        destination = parent / f"analysis_{analysis_id}"
        try:
            destination.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return analysis_id, destination
    raise FileExistsError("could not allocate a unique UUID analysis directory")


def code_provenance(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in paths:
        source = Path(item).resolve()
        records.append(
            {
                "path": str(source),
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
            }
        )
    records.sort(key=lambda record: record["path"])
    return records


def state_points(
    manifest: Mapping[str, Any], manifest_path: str | Path | None = None
) -> list[dict[str, Any]]:
    candidates = manifest.get("state_points")
    if candidates is None and isinstance(manifest.get("outputs"), Mapping):
        candidates = manifest["outputs"].get("state_points")
    if candidates is None and manifest.get("stage_manifests") is not None:
        if manifest_path is None:
            raise ManifestSchemaError(
                "manifest_path is required to resolve stage_manifests"
            )
        candidates = _state_points_from_stage_manifests(manifest, manifest_path)
    if not isinstance(candidates, list):
        raise ManifestSchemaError(
            "run manifest must contain state_points or verified stage_manifests"
        )
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ManifestSchemaError(f"state_points[{index}] must be an object")
        state_id = state_point_id(candidate)
        if state_id in seen:
            raise ManifestSchemaError(f"duplicate state_point_id: {state_id}")
        seen.add(state_id)
        result.append(candidate)
    return result


def _state_points_from_stage_manifests(
    manifest: Mapping[str, Any], manifest_path: str | Path
) -> list[dict[str, Any]]:
    """Adapt finalized stage bundles to the analysis-neutral state-point view."""

    references = manifest.get("stage_manifests")
    if not isinstance(references, list) or not references:
        raise ManifestSchemaError("stage_manifests must be a non-empty array")
    derived: list[dict[str, Any]] = []
    for index, reference in enumerate(references):
        stage_manifest_artifact = _verify_artifact(
            manifest_path,
            manifest,
            reference,
            require_declared_hash=True,
        )
        stage_manifest_path = Path(stage_manifest_artifact["path"])
        stage_manifest = load_json(stage_manifest_path)
        raw_artifacts = stage_manifest.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise ManifestSchemaError(
                f"stage manifest lacks artifacts: {stage_manifest_path}"
            )
        spec_reference = _artifact_with_suffix(raw_artifacts, "stage_spec.json")
        stage_spec_artifact = _verify_artifact(
            stage_manifest_path,
            stage_manifest,
            spec_reference,
            require_declared_hash=True,
        )
        stage_spec = load_json(stage_spec_artifact["path"])
        raw_state = stage_spec.get("state")
        state = raw_state if isinstance(raw_state, Mapping) else stage_spec
        stage_kind = str(
            state.get("stage_kind", stage_spec.get("stage_kind", ""))
        ).lower()
        if stage_kind != "production":
            # Initialization and equilibration are provenance, never implicit
            # density or Tg observations.
            continue
        production_reference = _production_thermo_artifact(raw_artifacts)
        production_artifact = _verify_artifact(
            stage_manifest_path,
            stage_manifest,
            production_reference,
            require_declared_hash=True,
        )
        physical_state_id = state.get(
            "state_id", stage_spec.get("state_id", stage_manifest.get("stage_id"))
        )
        if not isinstance(physical_state_id, str) or not physical_state_id:
            raise ManifestSchemaError(
                f"stage spec lacks state.state_id: {stage_spec_artifact['path']}"
            )
        replica_id = str(
            stage_spec.get("replica_id", stage_manifest.get("replica_id", ""))
        )
        stage_id = str(
            stage_manifest.get(
                "stage_id", stage_spec.get("stage_id", f"stage_{index:04d}")
            )
        )
        analysis_state_id = stage_id
        if not analysis_state_id or any(
            item.get("state_point_id") == analysis_state_id for item in derived
        ):
            analysis_state_id = f"{replica_id}::{physical_state_id}::{index:04d}"

        temperature_start = _state_value(
            state,
            stage_spec,
            ("temperature_start_k", "temperature_start_K"),
            "temperature_start_k",
        )
        temperature_end = _state_value(
            state,
            stage_spec,
            ("temperature_end_k", "temperature_end_K"),
            "temperature_end_k",
        )
        pressure_start = _state_value(
            state,
            stage_spec,
            ("pressure_start_bar",),
            "pressure_start_bar",
        )
        pressure_end = _state_value(
            state,
            stage_spec,
            ("pressure_end_bar",),
            "pressure_end_bar",
        )
        is_hold = temperature_start == temperature_end and pressure_start == pressure_end
        raw_roles = stage_spec.get("roles")
        if isinstance(raw_roles, str):
            roles = [raw_roles]
        elif isinstance(raw_roles, Sequence):
            roles = [str(role) for role in raw_roles]
        else:
            roles = []
            if bool(state.get("use_for_tg_fit", stage_spec.get("use_for_tg_fit"))):
                roles.append("tg_fit")
            if bool(state.get("use_for_density", stage_spec.get("use_for_density"))):
                roles.append("density_target")
        branch = stage_spec.get("branch", state.get("branch"))
        if branch is None and manifest.get("thermal_mode") == "tg":
            # thermal-properties resolved Tg specs admit only the cooling branch.
            branch = "cooling"
        execution_status = str(stage_manifest.get("execution_status", "")).upper()
        qc_status = str(stage_manifest.get("qc_status", "")).upper()
        derived.append(
            {
                "state_point_id": analysis_state_id,
                "physical_state_id": physical_state_id,
                "replica_id": replica_id,
                "status": execution_status,
                "validated": execution_status == "COMPLETE",
                "qc_status": qc_status,
                "roles": roles,
                "branch": branch,
                "mode": "hold" if is_hold else "ramp",
                "ensemble": "npt",
                "target_temperature_K": temperature_start,
                "target_pressure_bar": pressure_start,
                "temperature_end_K": temperature_end,
                "pressure_end_bar": pressure_end,
                "artifacts": {
                    "thermo_samples": {
                        "path": production_artifact["path"],
                        "sha256": production_artifact["actual_sha256"],
                        "bytes": production_artifact["bytes"],
                        "phase": "production",
                        "role": "thermo_samples",
                    }
                },
                "_source_layout": "stage_manifests",
                "_lineage_artifacts": [
                    stage_manifest_artifact,
                    stage_spec_artifact,
                    production_artifact,
                ],
            }
        )
    return derived


def state_point_id(state_point: Mapping[str, Any]) -> str:
    value = state_point.get(
        "state_point_id", state_point.get("state_id", state_point.get("id"))
    )
    if not isinstance(value, str) or not value.strip():
        raise ManifestSchemaError("every state point needs a non-empty state_point_id")
    return value


def state_point_target_temperature(state_point: Mapping[str, Any]) -> float:
    for key in (
        "target_temperature_K",
        "target_temperature_k",
        "temperature_K",
        "temperature_k",
        "temperature_start_k",
        "T_set_K",
    ):
        if key in state_point:
            return _finite_float(state_point[key], f"state point {key}")
    target = state_point.get("target")
    if isinstance(target, Mapping):
        for key in ("temperature_K", "T_K"):
            if key in target:
                return _finite_float(target[key], f"state point target.{key}")
    raise ManifestSchemaError(
        f"state point {state_point_id(state_point)} lacks a target temperature"
    )


def state_point_target_pressure(state_point: Mapping[str, Any]) -> float:
    for key in (
        "target_pressure_bar",
        "pressure_bar",
        "pressure_start_bar",
        "P_set_bar",
    ):
        if key in state_point:
            return _finite_float(state_point[key], f"state point {key}")
    target = state_point.get("target")
    if isinstance(target, Mapping):
        for key in ("pressure_bar", "P_bar"):
            if key in target:
                return _finite_float(target[key], f"state point target.{key}")
    raise ManifestSchemaError(
        f"state point {state_point_id(state_point)} lacks a target pressure"
    )


def state_point_branch(
    state_point: Mapping[str, Any], manifest: Mapping[str, Any]
) -> str | None:
    value = state_point.get("branch")
    if value is None and isinstance(manifest.get("protocol"), Mapping):
        value = manifest["protocol"].get("branch")
    if value is None:
        return None
    return str(value).strip().lower()


def state_point_roles(state_point: Mapping[str, Any]) -> list[str]:
    roles = state_point.get("roles", [])
    if isinstance(roles, str):
        roles = [roles]
    if not isinstance(roles, Sequence):
        raise ManifestSchemaError(
            f"roles for {state_point_id(state_point)} must be a string or array"
        )
    result = [str(role).strip().lower() for role in roles]
    if not result:
        if state_point.get("use_for_tg_fit") is True:
            result.append("tg_fit")
        if state_point.get("use_for_density") is True:
            result.append("density_target")
    return result


def run_is_finalized(manifest: Mapping[str, Any]) -> bool:
    return manifest.get("finalized") is True


def run_status(manifest: Mapping[str, Any]) -> str:
    status = manifest.get("status", manifest.get("simulation_status", ""))
    if isinstance(status, Mapping):
        status = status.get("state", status.get("status", ""))
    return str(status).strip().upper()


def state_point_validation_errors(
    state_point: Mapping[str, Any], manifest: Mapping[str, Any]
) -> list[str]:
    """Return fail-closed finalization errors for one reusable state point."""

    errors: list[str] = []
    if not run_is_finalized(manifest):
        errors.append("RUN_UNFINALIZED")
    if run_status(manifest) not in {
        "COMPLETE",
        "COMPLETE_WITH_QC_FAILURE",
    }:
        errors.append("RUN_NOT_REUSABLE")
    status = str(
        state_point.get("status", state_point.get("execution_status", ""))
    ).strip().upper()
    if status != "COMPLETE":
        errors.append("STATEPOINT_UNFINALIZED")
    if state_point.get("validated") is not True:
        errors.append("STATEPOINT_UNVALIDATED")
    qc = state_point.get("qc_status", state_point.get("qc", ""))
    if isinstance(qc, Mapping):
        qc = qc.get("status", "")
    if str(qc).strip().upper() != "PASS":
        errors.append("SOURCE_QC_FAILED")
    return errors


def manifest_total_mass_amu(
    manifest: Mapping[str, Any], state_point: Mapping[str, Any] | None = None
) -> float | None:
    containers: list[Mapping[str, Any]] = []
    if state_point is not None:
        containers.append(state_point)
    system = manifest.get("system")
    if isinstance(system, Mapping):
        containers.append(system)
    containers.append(manifest)
    for container in containers:
        for key in ("total_mass_amu", "total_mass_u", "mass_amu"):
            if key in container:
                return _finite_float(container[key], key)
    return None


def state_point_expected_atom_count(
    manifest: Mapping[str, Any], state_point: Mapping[str, Any]
) -> int | None:
    """Resolve the authenticated atom count associated with one replica."""

    direct = state_point.get("expected_atom_count")
    if isinstance(direct, int) and not isinstance(direct, bool) and direct > 0:
        return direct
    replica_id = str(state_point.get("replica_id", ""))
    system = manifest.get("system")
    if isinstance(system, Mapping):
        snapshots = system.get("input_snapshots")
        if isinstance(snapshots, Mapping):
            record = snapshots.get(replica_id)
            if isinstance(record, Mapping):
                value = record.get("atom_count")
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    return value
        value = system.get("atom_count")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def state_point_packing_id(state_point: Mapping[str, Any]) -> str | None:
    value = state_point.get("packing_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def load_state_point_samples(
    manifest_path: str | Path,
    manifest: Mapping[str, Any],
    state_point: Mapping[str, Any],
    *,
    require_declared_hash: bool = True,
) -> LoadedSamples:
    """Load one state point without changing or normalizing the source file."""

    state_id = state_point_id(state_point)
    artifact_ref = _find_thermo_artifact(manifest, state_point)
    artifact = _verify_artifact(
        manifest_path,
        manifest,
        artifact_ref,
        require_declared_hash=require_declared_hash,
    )
    try:
        frame = pd.read_csv(artifact["path"], compression="infer")
    except Exception as exc:  # pandas exposes parser and compression exceptions
        raise AnalysisInputError(
            f"cannot read thermo samples for {state_id}: {exc}",
            reason_code="SOURCE_ARTIFACT_UNREADABLE",
        ) from exc

    if "state_point_id" in frame.columns:
        selected = frame.loc[frame["state_point_id"].astype(str) == state_id].copy()
        if selected.empty:
            raise AnalysisInputError(
                f"artifact contains no rows for state point {state_id}",
                reason_code="STATEPOINT_ABSENT",
            )
        frame = selected
    else:
        frame = frame.copy()
        frame.insert(0, "state_point_id", state_id)

    missing = [column for column in REQUIRED_THERMO_COLUMNS if column not in frame]
    artifact_role = (
        artifact_ref.get("role", artifact_ref.get("name", ""))
        if isinstance(artifact_ref, Mapping)
        else ""
    )
    artifact_phase = (
        artifact_ref.get("phase", "") if isinstance(artifact_ref, Mapping) else ""
    )
    declared_phase = state_point.get(
        "_artifact_phase", state_point.get("artifact_phase", "")
    )
    production_semantics = (
        str(declared_phase).lower() == "production"
        or str(artifact_phase).lower() == "production"
        or str(artifact_role).lower()
        in {
            "production",
            "production_samples",
            "thermo_production_samples",
        }
    )
    if "phase" in missing and production_semantics:
        frame["phase"] = "production"
        missing.remove("phase")
    if missing:
        raise ManifestSchemaError(
            f"thermo artifact for {state_id} lacks columns: {', '.join(missing)}"
        )
    numeric_columns = (
        "step",
        "time_ps",
        "temp_K",
        "press_bar",
        "volume_A3",
        "density_g_cm3",
        "pe_eV",
        "ke_eV",
        "etotal_eV",
        "enthalpy_eV",
        "fmax_eV_A",
        "lx_A",
        "ly_A",
        "lz_A",
        "atom_count",
    )
    for column in numeric_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    total_mass_amu = manifest_total_mass_amu(manifest, state_point)
    if total_mass_amu is not None:
        volume = frame["volume_A3"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            recomputed = total_mass_amu * AMU_PER_ANGSTROM3_TO_G_CM3 / volume
        frame["_recomputed_density_g_cm3"] = recomputed
        if "density_g_cm3" not in frame:
            frame["density_g_cm3"] = recomputed
    elif "density_g_cm3" not in frame:
        raise ManifestSchemaError(
            f"{state_id} needs density_g_cm3 samples or system.total_mass_amu"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        frame["specific_volume_cm3_g"] = 1.0 / frame["density_g_cm3"].to_numpy(
            dtype=float
        )
    if isinstance(state_point.get("_lineage_artifacts"), list):
        artifact["lineage"] = state_point["_lineage_artifacts"]
    return LoadedSamples(state_id, frame, artifact)


def production_samples(frame: pd.DataFrame, state_id: str) -> pd.DataFrame:
    phases = frame["phase"].astype(str).str.strip().str.lower()
    result = frame.loc[phases == "production"].copy()
    if result.empty:
        raise AnalysisInputError(
            f"state point {state_id} has no production rows",
            reason_code="INSUFFICIENT_SAMPLES",
        )
    return result


def _run_root(manifest_path: Path, manifest: Mapping[str, Any]) -> Path:
    declared = manifest.get("run_dir", manifest.get("run_root"))
    if isinstance(declared, str) and declared:
        root = Path(declared).expanduser()
        if not root.is_absolute():
            # Canonical final manifests live at ``RUN/manifest/run_manifest.json``
            # while their artifact paths and ``run_dir: \".\"`` are relative to
            # RUN.  Preserve the ordinary same-directory interpretation for
            # manifests that are not stored in a dedicated manifest directory.
            anchor = (
                manifest_path.parent.parent
                if root == Path(".") and manifest_path.parent.name in {"manifest", "meta"}
                else manifest_path.parent
            )
            root = anchor / root
        return root.resolve()
    parent = manifest_path.resolve().parent
    if parent.name in {"manifest", "meta"}:
        return parent.parent
    return parent


def _find_thermo_artifact(
    manifest: Mapping[str, Any], state_point: Mapping[str, Any]
) -> str | Mapping[str, Any]:
    state_id = state_point_id(state_point)
    local = state_point.get("artifacts")
    if isinstance(local, Mapping):
        for key in (
            "production_samples",
            "thermo_production_samples",
            "thermo_samples",
            "samples",
            "thermo",
        ):
            if key in local:
                reference = local[key]
                if key in {"production_samples", "thermo_production_samples"} and isinstance(
                    reference, Mapping
                ):
                    reference = dict(reference)
                    reference.setdefault("role", "production_samples")
                return reference
    for key in (
        "production_artifact",
        "production_samples",
        "thermo_samples",
        "samples",
        "thermo_artifact",
    ):
        if key in state_point:
            reference = state_point[key]
            if key in {"production_artifact", "production_samples"} and isinstance(
                reference, Mapping
            ):
                reference = dict(reference)
                reference.setdefault("role", "production_samples")
            return reference

    global_artifacts = manifest.get("artifacts")
    if isinstance(global_artifacts, Mapping):
        for key in ("thermo_samples", "samples", "thermo"):
            candidate = global_artifacts.get(key)
            if isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, Mapping) and str(
                        item.get("state_point_id", "")
                    ) == state_id:
                        return item
            elif candidate is not None:
                return candidate
    elif isinstance(global_artifacts, list):
        for item in global_artifacts:
            if not isinstance(item, Mapping):
                continue
            role = str(item.get("role", item.get("name", ""))).lower()
            item_state = str(item.get("state_point_id", ""))
            if role in {"thermo_samples", "samples", "thermo"} and item_state in {
                "",
                state_id,
            }:
                return item
    raise ManifestSchemaError(f"no thermo-sample artifact declared for {state_id}")


def _verify_artifact(
    manifest_path: str | Path,
    manifest: Mapping[str, Any],
    artifact_ref: str | Mapping[str, Any],
    *,
    require_declared_hash: bool,
) -> dict[str, Any]:
    if isinstance(artifact_ref, str):
        raw_path = artifact_ref
        declared_hash = None
    elif isinstance(artifact_ref, Mapping):
        raw_path = artifact_ref.get(
            "path", artifact_ref.get("relative_path", artifact_ref.get("uri"))
        )
        declared_hash = artifact_ref.get("sha256")
    else:
        raise ManifestSchemaError("artifact reference must be a path or object")
    if not isinstance(raw_path, str) or not raw_path:
        raise ManifestSchemaError("artifact reference lacks a path")
    if require_declared_hash and not (
        isinstance(declared_hash, str) and len(declared_hash) == 64
    ):
        raise ArtifactIntegrityError(
            f"artifact {raw_path} lacks a declared SHA256",
            reason_code="ARTIFACT_HASH_MISSING",
        )

    manifest_source = Path(manifest_path).resolve()
    root = _run_root(manifest_source, manifest)
    source = Path(raw_path).expanduser()
    if not source.is_absolute():
        source = root / source
    source = source.resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ArtifactIntegrityError(
            f"artifact escapes run root: {source}", reason_code="ARTIFACT_PATH_ESCAPE"
        ) from exc
    if not source.is_file():
        raise ArtifactIntegrityError(
            f"artifact does not exist: {source}", reason_code="SOURCE_ARTIFACT_MISSING"
        )
    declared_bytes = (
        artifact_ref.get("bytes", artifact_ref.get("size_bytes"))
        if isinstance(artifact_ref, Mapping)
        else None
    )
    if declared_bytes is not None and source.stat().st_size != int(declared_bytes):
        raise ArtifactIntegrityError(
            f"artifact size mismatch: {source}", reason_code="ARTIFACT_SIZE_MISMATCH"
        )
    actual_hash = sha256_file(source)
    if declared_hash is not None and actual_hash.lower() != str(declared_hash).lower():
        raise ArtifactIntegrityError(
            f"artifact SHA256 mismatch: {source}", reason_code="HASH_MISMATCH"
        )
    verified = {
        "path": str(source),
        "relative_path": str(source.relative_to(root)),
        "declared_sha256": declared_hash,
        "actual_sha256": actual_hash,
        "bytes": source.stat().st_size,
    }
    if isinstance(artifact_ref, Mapping):
        for semantic_key in ("role", "phase", "state_point_id"):
            if semantic_key in artifact_ref:
                verified[semantic_key] = artifact_ref[semantic_key]
    return verified


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ManifestSchemaError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ManifestSchemaError(f"{label} must be finite")
    return result


def _artifact_with_suffix(
    artifacts: Sequence[Any], suffix: str
) -> Mapping[str, Any]:
    normalized_suffix = suffix.replace("\\", "/")
    matches = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping)
        and str(artifact.get("path", "")).replace("\\", "/").endswith(
            normalized_suffix
        )
    ]
    if len(matches) != 1:
        raise ManifestSchemaError(
            f"expected exactly one stage artifact ending in {suffix}, found {len(matches)}"
        )
    return matches[0]


def _production_thermo_artifact(
    artifacts: Sequence[Any],
) -> Mapping[str, Any]:
    """Select only an artifact whose hashed manifest declares its semantics."""

    matches = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping)
        and str(artifact.get("phase", "")).strip().lower() == "production"
        and str(artifact.get("role", artifact.get("name", ""))).strip().lower()
        in {"thermo_samples", "production_samples", "thermo_production_samples"}
    ]
    if len(matches) != 1:
        raise ManifestSchemaError(
            "expected exactly one stage artifact declaring phase=production "
            f"and a thermo-sample role, found {len(matches)}"
        )
    return matches[0]


def _state_value(
    state: Mapping[str, Any],
    stage_spec: Mapping[str, Any],
    keys: Sequence[str],
    label: str,
) -> float:
    for container in (state, stage_spec):
        for key in keys:
            if key in container:
                return _finite_float(container[key], label)
    raise ManifestSchemaError(f"stage spec lacks resolved {label}")
