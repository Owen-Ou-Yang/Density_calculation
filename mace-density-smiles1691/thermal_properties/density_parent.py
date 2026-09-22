"""Explicit, read-only intake of one completed density-transition endpoint.

The parent is an initial condition, not a reusable qualified density result.
Its existing QC (including FAIL) is preserved; the new PILOT must independently
pass the unchanged transition policy before its density branch may start.
"""

from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .config import ThermalConfigError
from .provenance import ProvenanceError, canonical_sha256, sha256_file


_KEYS = {
    "run_root", "run_id", "replica_id", "run_spec_sha256",
    "stage_manifest_sha256", "restart_sha256", "expected_step",
}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
# The fixed selection in pilots/render_density_parallel15_20260902.py plus
# the original P020001 pilot. Keep runtime intake independent of renderer files.
_ALLOWED_POLYMER_IDS = frozenset({
    "P010001", "P010002", "P010008", "P010014", "P010072",
    "P020001", "P030024", "P040002", "P040048", "P040049",
    "P040054", "P040073", "P050002", "P070466", "P100005", "P100009",
})


def _fail(message: str) -> None:
    raise ThermalConfigError(f"density parent: {message}")


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be an object")
    return value


def _absolute_path(value: object, field: str, *, file: bool = False) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value or not Path(value).is_absolute():
        _fail(f"{field} must be an absolute path")
    if any(part in {".", ".."} for part in value.split("/")):
        _fail(f"{field} cannot contain dot path components")
    candidate = Path(value)
    for component in (*reversed(candidate.parents), candidate):
        if component.is_symlink():
            _fail(f"{field} contains a symlink: {component}")
    if file and (not candidate.is_file() or candidate.stat().st_size <= 0):
        _fail(f"{field} is not a nonempty regular file: {candidate}")
    return candidate


def _integer(value: object, field: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        _fail(f"{field} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def _number(value: object, field: str) -> Decimal:
    if isinstance(value, bool):
        _fail(f"{field} must be finite numeric")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        _fail(f"{field} must be finite numeric")
    if not result.is_finite():
        _fail(f"{field} must be finite numeric")
    return result


def _load(path: Path) -> dict[str, Any]:
    _absolute_path(str(path), "artifact", file=True)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        canonical_sha256(value)  # Reject NaN/Infinity before trusting a document.
    except (OSError, ValueError, TypeError) as exc:
        _fail(f"cannot read canonical JSON {path}: {exc}")
    return dict(_mapping(value, str(path)))


def _file(path: Path) -> dict[str, Any]:
    _absolute_path(str(path), "artifact", file=True)
    try:
        return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    except (OSError, ProvenanceError) as exc:
        _fail(f"cannot hash parent artifact {path}: {exc}")


def _equal(left: object, right: object, field: str) -> None:
    if canonical_sha256(left) != canonical_sha256(right):
        _fail(f"{field} mismatch")


def normalize_density_parent(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the explicit parent selector, without creating output."""
    parent = dict(_mapping(value, "selector"))
    if set(parent) != _KEYS:
        _fail(f"selector fields mismatch: missing={sorted(_KEYS - set(parent))}, "
              f"extra={sorted(map(str, set(parent) - _KEYS))}")
    root = _absolute_path(parent["run_root"], "run_root")
    for field in ("run_id", "replica_id"):
        if not isinstance(parent[field], str) or not _IDENTIFIER.fullmatch(parent[field]):
            _fail(f"{field} is not a safe identifier")
    if root.name not in {parent["run_id"], parent["run_id"] + ".incomplete"}:
        _fail("run_root basename does not match run_id")
    for field in ("run_spec_sha256", "stage_manifest_sha256", "restart_sha256"):
        if not isinstance(parent[field], str) or not _SHA256.fullmatch(parent[field]):
            _fail(f"{field} must be a lowercase SHA256")
    _integer(parent["expected_step"], "expected_step", positive=True)
    parent["run_root"] = str(root)
    return parent


def verify_density_parent(
    parent: Mapping[str, Any],
    resolved: Mapping[str, Any],
    execution_identity: Mapping[str, Any],
    run_id: str | None = None,
    run_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify pinned parent core evidence and its compatibility with a new run.

    ``run_root``, when supplied, is the new attempt directory, not the shared
    output base. Nothing is written and no model/system subprocess is invoked.
    Old checkpoint collections are not imported or reclassified as new work.
    """
    selector = normalize_density_parent(parent)
    root = Path(selector["run_root"])
    if not root.is_dir():
        _fail(f"run_root is not a directory: {root}")
    if run_id == selector["run_id"]:
        _fail("new run_id cannot reuse the parent run_id")
    if run_root is not None:
        destination = _absolute_path(str(run_root), "new run_root")
        if destination == root or destination in root.parents or root in destination.parents:
            _fail("new output and parent run_root must not overlap")
    if (resolved.get("thermal_mode") != "density" or resolved.get("run_class") != "PILOT"
            or resolved.get("scientific_eligible") is not False
            or resolved.get("reuse_eligible") is not False):
        _fail("intake requires a new nonreusable PILOT density run")
    replicas = resolved.get("replicas")
    if not isinstance(replicas, list) or len(replicas) != 1:
        _fail("intake requires exactly one replica")
    current_replica = _mapping(replicas[0], "new replica")
    reuse = resolved.get("reuse")
    if reuse is not None:
        reuse = _mapping(reuse, "reuse")
        if reuse.get("search_roots") or reuse.get("selected_state_point_provenance"):
            _fail("parent intake cannot be combined with cross-run result reuse")

    stage_root = root / "replicas" / selector["replica_id"] / "mace_transition_npt"
    paths = {
        "run_spec": root / "spec" / "run_spec.json",
        "stage_spec": stage_root / "stage_spec.json",
        "stage_manifest": stage_root / "stage_manifest.json",
        "qc": stage_root / "qc" / "state_point_qc.json",
        "production_samples": stage_root / "samples" / "production.csv",
        "restart": stage_root / "restart.final",
    }
    source_files = {key: _file(path) for key, path in paths.items()}
    for key, pin in (("run_spec", "run_spec_sha256"),
                     ("stage_manifest", "stage_manifest_sha256"),
                     ("restart", "restart_sha256")):
        if source_files[key]["sha256"] != selector[pin]:
            _fail(f"{key} SHA256 mismatch")
    parent_spec = _load(paths["run_spec"])
    resolved_hash = parent_spec.get("resolved_spec_sha256")
    identity_document = dict(parent_spec)
    identity_document.pop("resolved_spec_sha256", None)
    if canonical_sha256(identity_document) != resolved_hash:
        _fail("parent run_spec embedded resolved_spec_sha256 mismatch")
    stage_spec = _load(paths["stage_spec"])
    manifest = _load(paths["stage_manifest"])
    qc = _load(paths["qc"])
    if manifest.get("execution_status") != "COMPLETE":
        _fail("parent transition execution is not COMPLETE")
    parent_qc = manifest.get("qc_status")
    if parent_qc not in {"PASS", "FAIL"} or qc.get("status") != parent_qc:
        _fail("parent QC status/artifact mismatch")
    expected_stage = selector["replica_id"] + "__mace_transition_npt"
    for document in (stage_spec, manifest):
        if (document.get("stage_id") != expected_stage
                or document.get("replica_id") != selector["replica_id"]):
            _fail("parent stage/replica identity mismatch")
    if (stage_spec.get("stage_role") != "mace_transition_npt"
            or stage_spec.get("initialize_velocities") is not False):
        _fail("parent is not a noninitializing MACE transition")
    if canonical_sha256(stage_spec) != manifest.get("stage_spec_sha256"):
        _fail("parent stage_spec canonical hash mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        _fail("parent manifest lacks artifact inventory")
    for key in ("stage_spec", "qc", "production_samples", "restart"):
        relative = paths[key].relative_to(stage_root).as_posix()
        records = [entry for entry in artifacts
                   if isinstance(entry, Mapping) and entry.get("path") == relative]
        if len(records) != 1:
            _fail(f"parent manifest must bind exactly one {relative}")
        record = records[0]
        if (record.get("sha256") != source_files[key]["sha256"]
                or type(record.get("bytes")) is not int
                or record.get("bytes") != source_files[key]["bytes"]):
            _fail(f"parent manifest artifact hash/bytes mismatch: {relative}")

    parent_identity = _mapping(parent_spec.get("execution_identity"), "parent execution_identity")
    _equal(stage_spec.get("execution_identity"), parent_identity, "parent run/stage execution_identity")
    old_model = _mapping(parent_identity.get("model"), "parent model")
    new_model = _mapping(execution_identity.get("model"), "current model")
    for field in ("path", "sha256", "bytes", "head", "dtype"):
        if field not in old_model or field not in new_model:
            _fail(f"model lacks {field}")
        _equal(old_model[field], new_model[field], f"model.{field}")
    old_system = _mapping(parent_spec.get("system"), "parent system")
    new_system = _mapping(resolved.get("system"), "current system")
    for field in ("polymer_id", "mace_elements", "element_list", "mace_head", "mace_dtype"):
        if field not in old_system or field not in new_system:
            _fail(f"system lacks {field}")
        _equal(old_system[field], new_system[field], f"system.{field}")
    if (parent_spec.get("thermal_mode") != "density"
            or old_system.get("polymer_id") not in _ALLOWED_POLYMER_IDS
            or old_model.get("head") != "omol" or old_model.get("dtype") != "float32"):
        _fail("parent intake is limited to the fixed 15-polymer selection plus P020001 "
              "omol float32 density pilots")
    for system, model in ((old_system, old_model), (new_system, new_model)):
        for source, target in (("mace_model", "path"), ("mace_head", "head"), ("mace_dtype", "dtype")):
            _equal(system.get(source), model.get(target), f"system/model {source}")
    old_snapshots = _mapping(parent_identity.get("input_snapshots"), "parent input_snapshots")
    new_snapshots = _mapping(execution_identity.get("input_snapshots"), "current input_snapshots")
    old_snapshot = _mapping(old_snapshots.get(selector["replica_id"]), "parent replica snapshot")
    new_snapshot = _mapping(new_snapshots.get(current_replica.get("replica_id")), "current replica snapshot")
    snapshot_fields = ("snapshot_sha256", "metadata_sha256", "snapshot_class", "packing_id",
                       "atom_count", "atom_type_count", "element_counts")
    for field in snapshot_fields:
        if field not in old_snapshot or field not in new_snapshot:
            _fail(f"snapshot lacks {field}")
        _equal(old_snapshot[field], new_snapshot[field], f"snapshot.{field}")
    for field in ("engine", "initialize"):
        _mapping(parent_spec.get(field), f"parent {field}")
        _mapping(resolved.get(field), f"current {field}")
    for field in ("name", "device", "dt_ps", "tdamp_ps", "pdamp_ps"):
        old_value = parent_spec["engine"].get(field)
        new_value = resolved["engine"].get(field)
        if field.endswith("_ps"):
            if _number(old_value, field) != _number(new_value, field):
                _fail(f"engine.{field} mismatch")
        else:
            _equal(old_value, new_value, f"engine.{field}")
    old_engine = _mapping(parent_identity.get("engine"), "parent engine identity")
    new_engine = _mapping(execution_identity.get("engine"), "current engine identity")
    for field in ("name", "device", "runtime_dependency_files"):
        if field not in old_engine or field not in new_engine:
            _fail(f"engine identity lacks {field}")
        _equal(old_engine[field], new_engine[field], f"engine identity {field}")
    if not old_engine["runtime_dependency_files"]:
        _fail("parent must bind actual LAMMPS/MPI dependency identities")
    state = _mapping(stage_spec.get("state"), "parent transition state")
    target_temp = _number(resolved["initialize"].get("temperature_k"), "target temperature")
    target_press = _number(resolved["initialize"].get("mace_transition_pressure_bar"), "target pressure")
    for field in ("temperature_start_k", "temperature_end_k"):
        if _number(state.get(field), field) != target_temp:
            _fail("parent/current transition temperature mismatch")
    for field in ("pressure_start_bar", "pressure_end_bar"):
        if _number(state.get(field), field) != target_press:
            _fail("parent/current transition pressure mismatch")
    if target_temp != Decimal("305"):
        _fail("parent intake must first validate the 305 K transition")
    if state.get("use_for_density") is not False or state.get("use_for_tg_fit") is not False:
        _fail("parent transition must not be declared a density or Tg result")

    start = _integer(stage_spec.get("start_step"), "parent start_step")
    equil = _integer(stage_spec.get("equilibration_steps"), "parent equilibration_steps")
    prod = _integer(stage_spec.get("production_steps"), "parent production_steps", positive=True)
    endpoint = start + equil + prod
    if endpoint != selector["expected_step"]:
        _fail("parent endpoint does not match expected_step")
    dt = _number(parent_spec["engine"].get("dt_ps"), "parent timestep")
    last: dict[str, str] | None = None
    previous_step: Decimal | None = None
    try:
        with paths["production_samples"].open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"step", "time_ps", "atom_count", "volume_A3", "density_g_cm3", "lx_A", "ly_A", "lz_A"}
            if not required.issubset(reader.fieldnames or []):
                _fail("parent production samples lack endpoint/structural columns")
            for row in reader:
                values = {key: _number(value, f"sample {key}") for key, value in row.items()}
                step = values["step"]
                if step != step.to_integral_value() or (previous_step is not None and step <= previous_step):
                    _fail("parent production sample steps must be increasing integers")
                if values["atom_count"] != old_snapshot["atom_count"]:
                    _fail("parent sample atom_count differs from snapshot")
                if any(values[field] <= 0 for field in ("volume_A3", "density_g_cm3", "lx_A", "ly_A", "lz_A")):
                    _fail("parent sample cell/density is invalid")
                previous_step, last = step, row
    except (OSError, csv.Error) as exc:
        _fail(f"cannot read parent production samples: {exc}")
    if last is None or _number(last["step"], "last step") != endpoint:
        _fail("parent production last step does not match endpoint")
    if abs(_number(last["time_ps"], "last time") - endpoint * dt) > Decimal("1e-9"):
        _fail("parent production last time does not match endpoint/timestep")
    merge = _mapping(manifest.get("segment_merge"), "parent segment_merge")
    merge_production = _mapping(merge.get("production"), "parent production merge")
    if merge_production.get("last_step") != endpoint or merge_production.get("sha256") != source_files["production_samples"]["sha256"]:
        _fail("parent merged production endpoint/hash mismatch")
    # Recheck bytes after interpretation so a source change cannot be copied as
    # though it were the evidence we just validated.
    for key, path in paths.items():
        if _file(path) != source_files[key]:
            _fail(f"parent artifact changed during validation: {key}")
    return {
        "schema_version": "thermal-properties-density-parent-intake/v1",
        "parent_run_id": selector["run_id"],
        "parent_replica_id": selector["replica_id"],
        "parent_stage_id": expected_stage,
        "parent_run_root": str(root),
        "parent_qc_status": parent_qc,
        "parent_execution_status": "COMPLETE",
        "parent_resolved_spec_sha256": resolved_hash,
        "start_step": endpoint,
        "start_time_ps": float(endpoint * dt),
        "source_files": source_files,
        "model": dict(old_model),
        "snapshot_lineage": {field: old_snapshot[field] for field in snapshot_fields},
        "parent_execution_identity": dict(parent_identity),
        "parent_transition_state": dict(state),
        "parent_is_qualified_density": False,
        "old_qc_modified": False,
    }
