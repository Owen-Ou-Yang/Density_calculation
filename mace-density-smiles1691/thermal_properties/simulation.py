"""Analysis-neutral planning and resumable execution for thermal NPT MD.

This module is deliberately independent of the modulus workflow.  It expands
the human-facing thermal configuration into explicit state points, records the
resolved plan, and provides a stage runner whose filesystem contract is safe to
resume.  Scientific analyses consume only finalized manifests and never call
the execution functions in this module.

The production runner is intentionally small: a caller supplies the LAMMPS
command and may inject a process runner for tests.  No calculation is launched
while importing or planning.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import errno
import fcntl
import hashlib
import functools
import json
import os
from pathlib import Path
import re
import signal
import shutil
import stat
import subprocess
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid

from .config import (
    StageKind,
    StatePointConfig,
    ThermalConfigError,
    ThermalProtocolConfig,
    ps_to_steps,
)
from .planning import PlanStep, ThermalPlan, build_thermal_plan
from .density_parent import normalize_density_parent, verify_density_parent
from .provenance import (
    AnalysisStatus,
    ArtifactProvenance,
    ExecutionStatus,
    ProvenanceError,
    QCStatus,
    StatePointProvenance,
    canonical_sha256,
    sha256_file,
    read_state_point_provenance,
    write_state_point_provenance,
)
from .reuse import (
    AmbiguousReuseError,
    ReuseAction,
    ReuseError,
    ReuseRequest,
    density_qc_implementation_sha256,
    resolve_cross_run_reuse,
    select_reuse_candidate,
)
from .snapshot_contract import (
    SnapshotClass,
    SnapshotContractError,
    load_snapshot_contract,
)
from .submission_claim import (
    CLAIM_ROOT_NAME,
    OWNER_MARKER_NAME,
    ClaimValidationError,
    OutputOwnership,
    validate_run_id,
)


RUN_SPEC_SCHEMA = "thermal-properties-resolved-run-spec/v1"
DIRECT_OWNER_ROOT_NAME = ".direct_run_owners"
RUN_MANIFEST_SCHEMA = "thermal-properties-run-manifest/v1"
STAGE_SPEC_SCHEMA = "thermal-properties-stage-spec/v1"
STAGE_MANIFEST_SCHEMA = "thermal-properties-stage-manifest/v1"
CHECKPOINT_SCHEMA = "thermal-properties-checkpoint/v1"
RUNTIME_INPUT_FILES = (
    "in.initialize_mace_mh1.lmp",
    "in.npt_stage_mace_mh1.lmp",
    "mace_mh1_thermal_setup.mod",
    "thermal_temperature.mod",
)
ORCHESTRATION_SOURCE_FILES = (
    "__init__.py",
    "cli.py",
    "config.py",
    "density_parent.py",
    "planning.py",
    "provenance.py",
    "reuse.py",
    "sampling_continuation.py",
    "simulation.py",
    "snapshot_contract.py",
    "submission_claim.py",
)
QUALITY_CONTROL_SOURCE_FILES = (
    "analysis/__init__.py",
    "analysis/convergence.py",
)
QUALITY_CONTROL_POLICY_FILES = (
    "config/tg_fit_v1.json",
    "config/density_qc_v1.json",
    "config/mace_transition_qc_v1.json",
)
RUN_CLASSES = {
    "ENGINEERING_SMOKE",
    "DEBUG",
    "PILOT",
    "PRODUCTION",
}
NONSCIENTIFIC_RUN_CLASSES = {"ENGINEERING_SMOKE", "DEBUG", "PILOT"}

# Intel MPI can leave the surviving rank waiting after its peer raises a
# Python-side CUDA OOM.  Detect the two concrete PyTorch OOM signatures in
# either output stream and terminate the whole process group so the
# normal stage failure path can persist an INCOMPLETE result promptly.
_FATAL_RUNTIME_STDERR_PATTERNS = (
    b"RuntimeError: CUDA out of memory",
    b"torch.OutOfMemoryError: CUDA out of memory",
)
_FATAL_RUNTIME_RETURN_CODE = 86
_FATAL_RUNTIME_POLL_SECONDS = 0.1
_FATAL_RUNTIME_TERMINATION_GRACE_SECONDS = 5.0
_FATAL_RUNTIME_KILL_GRACE_SECONDS = 2.0


class ThermalSimulationError(RuntimeError):
    """Raised when planning, execution, or artifact validation fails closed."""


class StageIntegrityError(ThermalSimulationError):
    """A supposedly reusable stage/checkpoint does not match its manifest."""


class SamplingNotConvergedError(ThermalSimulationError):
    """Completed MD windows stopped at an authenticated scientific gate."""

    def __init__(self, code: str, record_path: Path):
        self.code = code
        self.record_path = record_path.resolve()
        super().__init__(f"{code}: {self.record_path}")


def verify_sampling_outcome(path: str | Path) -> dict[str, Any]:
    from .sampling_continuation import verify_sampling_outcome as verify

    return verify(path)


class DuplicateSampleStepError(StageIntegrityError):
    """Two sample segments contain the same MD timestep."""


@dataclass(frozen=True)
class CheckpointRecord:
    """Hash-verified progress from an interrupted NPT state point."""

    path: Path
    sha256: str
    size_bytes: int
    completed_equilibration_steps: int
    completed_production_steps: int
    attempt_index: int
    sidecar_path: Path
    sidecar_sha256: str

    @property
    def completed_steps(self) -> int:
        return self.completed_equilibration_steps + self.completed_production_steps


@dataclass(frozen=True)
class StageInvocation:
    """Everything a process runner needs for exactly one execution segment."""

    command: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    stdout_path: Path
    stderr_path: Path
    log_path: Path
    equilibration_segment_path: Path
    production_segment_path: Path
    final_restart_path: Path
    checkpoint_dir: Path
    resume_checkpoint: CheckpointRecord | None
    remaining_equilibration_steps: int
    remaining_production_steps: int
    segment_index: int


@dataclass(frozen=True)
class StageExecutionResult:
    stage_id: str
    stage_dir: Path
    execution_status: str
    qc_status: str
    analysis_status: str
    skipped: bool
    resumed_from: str | None
    manifest_path: Path | None


ProcessRunner = Callable[[StageInvocation], int | subprocess.CompletedProcess[Any]]


def _decimal(value: object, field: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ThermalConfigError(f"{field} must be numeric, not bool")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ThermalConfigError(f"{field} must be a finite decimal number") from exc
    if not result.is_finite() or (positive and result <= 0):
        relation = " > 0" if positive else " finite"
        raise ThermalConfigError(f"{field} must be{relation}")
    return result


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThermalConfigError(f"{field} must be an object")
    return value


def _require_sequence(value: object, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ThermalConfigError(f"{field} must be an array")
    return value


def _validate_object_keys(
    mapping: Mapping[str, Any],
    field: str,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    keys = {str(key) for key in mapping}
    missing = sorted(required - keys)
    extra = sorted(keys - required - set(optional))
    if missing:
        raise ThermalConfigError(f"{field} is missing required fields: {missing}")
    if extra:
        raise ThermalConfigError(f"{field} contains unsupported fields: {extra}")


def _temperature_label(value: Decimal) -> str:
    normalized = format(value.normalize(), "f")
    return normalized.replace("-", "m").replace(".", "p")


def _strictly_decreasing(values: Sequence[Decimal]) -> bool:
    return all(left > right for left, right in zip(values, values[1:]))


def resolved_spec_sha256(value: Mapping[str, Any]) -> str:
    """Hash a resolved spec without trusting its embedded digest field."""

    identity = dict(value)
    identity.pop("resolved_spec_sha256", None)
    return canonical_sha256(identity)


def _load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ThermalConfigError(f"failed to read thermal config: {source}") from exc
    if not isinstance(payload, dict):
        raise ThermalConfigError("thermal config root must be an object")
    return payload


def _reject_melting_configuration(value: object, path: str = "config") -> None:
    forbidden = {
        "tm",
        "tm_k",
        "melting_temperature",
        "melting_temperature_k",
        "melting",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).strip().lower() in forbidden:
                raise ThermalConfigError(
                    f"{path}.{key} is unsupported: this workflow has no Tm mode"
                )
            _reject_melting_configuration(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_melting_configuration(child, f"{path}[{index}]")


def expand_thermal_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Expand a high-level Tg/density request into an immutable explicit plan.

    No temperature is synthesized.  The Tg fit-temperature list is an explicit
    subset of the cooling schedule.  An optional, separately labelled density
    target may reuse one exact cooling state.  Density mode creates only its
    requested isothermal-aged state point.
    """

    _reject_melting_configuration(payload)
    schema = payload.get("schema_version")
    if schema != "thermal_schema_v1":
        raise ThermalConfigError(
            "schema_version must be exactly 'thermal_schema_v1'"
        )
    mode = payload.get("thermal_mode")
    if mode not in {"tg", "density"}:
        raise ThermalConfigError("thermal_mode must be 'tg' or 'density'")

    _validate_object_keys(
        payload,
        "config",
        required={
            "schema_version",
            "thermal_mode",
            "run_class",
            "scientific_eligible",
            "reuse_eligible",
            "quality_tier",
            "system",
            "replicas",
            "engine",
            "initialize",
            "npt",
            "output_root",
            "scheduler",
            mode,
        },
        optional={"reuse", "density_parent_restart", "sampling_continuation"},
    )
    if payload.get("quality_tier") not in {"pilot", "production"}:
        raise ThermalConfigError("quality_tier must be 'pilot' or 'production'")
    run_class = payload.get("run_class")
    scientific_eligible = payload.get("scientific_eligible")
    reuse_eligible = payload.get("reuse_eligible")
    if run_class not in RUN_CLASSES:
        raise ThermalConfigError(
            "run_class must be ENGINEERING_SMOKE, DEBUG, PILOT, or PRODUCTION"
        )
    if type(scientific_eligible) is not bool or type(reuse_eligible) is not bool:
        raise ThermalConfigError(
            "scientific_eligible and reuse_eligible must be explicit booleans"
        )
    if run_class in NONSCIENTIFIC_RUN_CLASSES:
        if scientific_eligible or reuse_eligible:
            raise ThermalConfigError(
                f"{run_class} runs must set scientific_eligible=false and "
                "reuse_eligible=false"
            )
        if payload.get("quality_tier") != "pilot":
            raise ThermalConfigError(
                f"{run_class} runs require quality_tier='pilot'"
            )
    else:
        if scientific_eligible is not True:
            raise ThermalConfigError(
                "PRODUCTION runs require scientific_eligible=true"
            )
        if payload.get("quality_tier") != "production":
            raise ThermalConfigError(
                "PRODUCTION runs require quality_tier='production'"
            )
        if reuse_eligible and not scientific_eligible:
            raise ThermalConfigError(
                "reuse_eligible=true requires scientific_eligible=true"
            )
        if reuse_eligible:
            raise ThermalConfigError(
                "cross-run density reuse v1 cannot satisfy the production "
                "three-packing contract; set reuse_eligible=false and aggregate "
                "new or explicitly analyzed replica results"
            )
    if not isinstance(payload.get("output_root"), str) or not payload["output_root"]:
        raise ThermalConfigError("output_root must be a nonempty string")

    parent_restart = None
    sampling_continuation = None
    if "sampling_continuation" in payload:
        if (mode != "density" or run_class != "PILOT" or scientific_eligible
                or reuse_eligible or "reuse" in payload):
            raise ThermalConfigError("sampling_continuation requires nonreusable density PILOT")
        sampling = _require_mapping(payload["sampling_continuation"], "sampling_continuation")
        expected_sampling = {"increment_ps": 25.0, "max_transition_ps": 100.0,
                             "max_density_ps": 100.0}
        _validate_object_keys(sampling, "sampling_continuation", required=set(expected_sampling))
        for field, expected in expected_sampling.items():
            if _decimal(sampling[field], f"sampling_continuation.{field}") != Decimal(str(expected)):
                raise ThermalConfigError(f"sampling_continuation.{field} must be {expected}")
        sampling_continuation = expected_sampling
    if "density_parent_restart" in payload:
        if mode != "density" or run_class != "PILOT" or "reuse" in payload:
            raise ThermalConfigError(
                "density_parent_restart requires density PILOT without scientific reuse"
            )
        parent_restart = normalize_density_parent(
            _require_mapping(payload["density_parent_restart"], "density_parent_restart")
        )

    system = dict(_require_mapping(payload.get("system"), "system"))
    engine = dict(_require_mapping(payload.get("engine"), "engine"))
    initialize = dict(_require_mapping(payload.get("initialize"), "initialize"))
    npt = dict(_require_mapping(payload.get("npt"), "npt"))
    scheduler = _require_mapping(payload.get("scheduler"), "scheduler")
    reuse_config: dict[str, Any] | None = None
    if "reuse" in payload:
        if mode != "density":
            raise ThermalConfigError("reuse is supported only for density mode")
        reuse_mapping = _require_mapping(payload["reuse"], "reuse")
        _validate_object_keys(
            reuse_mapping,
            "reuse",
            required={"search_roots", "selected_state_point_provenance"},
        )
        search_roots = _require_sequence(
            reuse_mapping.get("search_roots"), "reuse.search_roots"
        )
        if not all(isinstance(item, str) and item for item in search_roots):
            raise ThermalConfigError("reuse.search_roots entries must be paths")
        selected = reuse_mapping.get("selected_state_point_provenance")
        if selected is not None and (not isinstance(selected, str) or not selected):
            raise ThermalConfigError(
                "reuse.selected_state_point_provenance must be null or a path"
            )
        reuse_config = {
            "search_roots": list(search_roots),
            "selected_state_point_provenance": selected,
        }
    _validate_object_keys(
        system,
        "system",
        required={
            "polymer_id",
            "input_data",
            "snapshot_metadata",
            "snapshot_class",
            "mace_model",
            "mace_head",
            "mace_dtype",
            "mace_elements",
            "element_list",
        },
    )
    _validate_object_keys(
        engine,
        "engine",
        required={
            "name",
            "device",
            "lammps_command",
            "runtime_dependencies",
            "dt_ps",
            "tdamp_ps",
            "pdamp_ps",
        },
    )
    _validate_object_keys(
        initialize,
        "initialize",
        required={
            "temperature_k",
            "minimize",
            "nvt_ps",
            "mace_transition_npt_ps",
            "mace_transition_pressure_bar",
        },
        optional={"min_etol", "min_ftol_eV_A", "min_maxiter", "min_maxeval"},
    )
    _validate_object_keys(
        npt,
        "npt",
        required={
            "pressure_bar",
            "ramp_ps",
            "equilibration_ps",
            "production_ps",
            "thermo_interval_ps",
            "sample_interval_ps",
            "restart_interval_ps",
        },
    )
    _validate_object_keys(
        scheduler,
        "scheduler",
        required={"submit_mode", "qsub_args"},
    )
    if scheduler.get("submit_mode") not in {"print", "qsub"}:
        raise ThermalConfigError("scheduler.submit_mode must be print or qsub")
    qsub_args = _require_sequence(scheduler.get("qsub_args"), "scheduler.qsub_args")
    if not all(isinstance(item, str) for item in qsub_args):
        raise ThermalConfigError("scheduler.qsub_args entries must be strings")
    if engine.get("name") != "mace_mh1":
        raise ThermalConfigError("engine.name must be exactly 'mace_mh1'")
    if engine.get("device") != "cuda":
        raise ThermalConfigError("engine.device must be exactly 'cuda'")
    command = _require_sequence(engine.get("lammps_command"), "engine.lammps_command")
    if not command or not all(isinstance(item, str) and item for item in command):
        raise ThermalConfigError("engine.lammps_command needs nonempty arguments")
    runtime_dependencies = _require_sequence(
        engine.get("runtime_dependencies"), "engine.runtime_dependencies"
    )
    if not all(
        isinstance(item, str) and item for item in runtime_dependencies
    ):
        raise ThermalConfigError(
            "engine.runtime_dependencies entries must be nonempty paths"
        )
    for field in ("dt_ps", "tdamp_ps", "pdamp_ps"):
        _decimal(engine.get(field), f"engine.{field}", positive=True)
    if not isinstance(initialize.get("minimize"), bool):
        raise ThermalConfigError("initialize.minimize must be boolean")
    _decimal(initialize.get("temperature_k"), "initialize.temperature_k", positive=True)
    for field in (
        "nvt_ps",
        "mace_transition_npt_ps",
        "min_etol",
        "min_ftol_eV_A",
    ):
        if field in initialize:
            _decimal(initialize[field], f"initialize.{field}", positive=True)
    _decimal(
        initialize.get("mace_transition_pressure_bar"),
        "initialize.mace_transition_pressure_bar",
    )
    for field in ("min_maxiter", "min_maxeval"):
        if field in initialize and (
            isinstance(initialize[field], bool)
            or not isinstance(initialize[field], int)
            or initialize[field] <= 0
        ):
            raise ThermalConfigError(f"initialize.{field} must be a positive integer")
    if not isinstance(system.get("polymer_id"), str) or not system["polymer_id"]:
        raise ThermalConfigError("system.polymer_id must be a nonempty string")
    for field in ("input_data", "snapshot_metadata", "mace_model"):
        if not isinstance(system.get(field), str) or not system[field]:
            raise ThermalConfigError(f"system.{field} must be a nonempty string")
    try:
        snapshot_class = SnapshotClass(system.get("snapshot_class"))
    except ValueError as exc:
        raise ThermalConfigError(
            "system.snapshot_class must be APG_RAW, CLASSICAL_EQ2, or "
            "MACE_EQUILIBRATED"
        ) from exc
    if snapshot_class is SnapshotClass.APG_RAW:
        raise ThermalConfigError(
            "APG_RAW is a dilute packing input for classical preprocessing, "
            "not a MACE Tg/density production snapshot"
        )
    elements = _require_sequence(system.get("mace_elements"), "system.mace_elements")
    if (
        not elements
        or not all(isinstance(item, str) and re.fullmatch(r"[A-Z][a-z]?", item) for item in elements)
    ):
        raise ThermalConfigError(
            "system.mace_elements must contain an ordered atom-type mapping "
            "of chemical symbols"
        )
    element_list = _require_sequence(system.get("element_list"), "system.element_list")
    if (
        not element_list
        or len(set(element_list)) != len(element_list)
        or not all(
            isinstance(item, str) and re.fullmatch(r"[A-Z][a-z]?", item)
            for item in element_list
        )
        or set(element_list) != set(elements)
    ):
        raise ThermalConfigError(
            "system.element_list must be the unique element set represented "
            "by system.mace_elements"
        )
    if system.get("mace_head") != "omol":
        raise ThermalConfigError("system.mace_head must be exactly 'omol'")
    if system.get("mace_dtype") != "float32":
        raise ThermalConfigError("system.mace_dtype must be exactly 'float32'")
    replicas = list(_require_sequence(payload.get("replicas"), "replicas"))
    if not replicas:
        raise ThermalConfigError("replicas must not be empty")

    timestep = _decimal(engine.get("dt_ps"), "engine.dt_ps", positive=True)
    ramp_ps = _decimal(npt.get("ramp_ps"), "npt.ramp_ps")
    if ramp_ps < 0:
        raise ThermalConfigError("npt.ramp_ps must be >= 0")
    equilibration_ps = _decimal(
        npt.get("equilibration_ps"), "npt.equilibration_ps", positive=True
    )
    production_ps = _decimal(
        npt.get("production_ps"), "npt.production_ps", positive=True
    )
    pressure_bar = _decimal(npt.get("pressure_bar"), "npt.pressure_bar")
    ramp_steps = ps_to_steps(ramp_ps, timestep) if ramp_ps > 0 else 0
    ps_to_steps(equilibration_ps, timestep)
    ps_to_steps(production_ps, timestep)
    interval_steps: dict[str, int] = {}
    for interval in (
        "thermo_interval_ps",
        "sample_interval_ps",
        "restart_interval_ps",
    ):
        interval_steps[interval] = ps_to_steps(npt.get(interval), timestep)
    initialization_steps = ps_to_steps(initialize.get("nvt_ps"), timestep)
    transition_steps = ps_to_steps(
        initialize.get("mace_transition_npt_ps"), timestep
    )
    sample_steps = interval_steps["sample_interval_ps"]
    for field, steps in (
        ("initialize.nvt_ps", initialization_steps),
        ("initialize.mace_transition_npt_ps", transition_steps),
        ("npt.ramp_ps", ramp_steps),
        ("npt.equilibration_ps", ps_to_steps(equilibration_ps, timestep)),
        ("npt.production_ps", ps_to_steps(production_ps, timestep)),
        ("npt.restart_interval_ps", interval_steps["restart_interval_ps"]),
    ):
        if steps and steps % sample_steps != 0:
            raise ThermalConfigError(
                f"{field} must be an integer multiple of npt.sample_interval_ps "
                "for gap-free resumable sample segments"
            )
    restart_steps = interval_steps["restart_interval_ps"]
    checkpointable_phases = [
        ("initialize.nvt_ps", initialization_steps),
        ("initialize.mace_transition_npt_ps", transition_steps),
        ("npt.equilibration_ps", ps_to_steps(equilibration_ps, timestep)),
        ("npt.production_ps", ps_to_steps(production_ps, timestep)),
    ]
    if ramp_steps:
        checkpointable_phases.append(("npt.ramp_ps", ramp_steps))
    if restart_steps > min(steps for _, steps in checkpointable_phases):
        raise ThermalConfigError(
            "npt.restart_interval_ps must not exceed initialize.nvt_ps, "
            "a nonzero npt.ramp_ps, npt.equilibration_ps, or "
            "npt.production_ps; every phase must "
            "be able to create a periodic checkpoint"
        )
    for field, steps in checkpointable_phases:
        if steps % restart_steps != 0:
            raise ThermalConfigError(
                f"{field} must be an integer multiple of "
                "npt.restart_interval_ps so its exact endpoint has a durable "
                "checkpoint"
            )

    replica_ids: list[str] = []
    replica_seeds: dict[str, int] = {}
    for index, item in enumerate(replicas):
        replica = _require_mapping(item, f"replicas[{index}]")
        _validate_object_keys(
            replica,
            f"replicas[{index}]",
            required={"replica_id", "seed"},
            optional={"input_data", "snapshot_metadata", "snapshot_class"},
        )
        replica_id = replica.get("replica_id")
        seed = replica.get("seed")
        if not isinstance(replica_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", replica_id
        ):
            raise ThermalConfigError(f"replicas[{index}].replica_id is required")
        if replica_id in replica_seeds:
            raise ThermalConfigError(f"duplicate replica_id: {replica_id}")
        if isinstance(seed, bool) or not isinstance(seed, int) or not (1 <= seed < 2147483647):
            raise ThermalConfigError(
                f"replicas[{index}].seed must be an integer in [1, 2147483646]"
            )
        replica_ids.append(replica_id)
        replica_seeds[replica_id] = seed

    if run_class == "PRODUCTION":
        if len(replicas) < 3:
            raise ThermalConfigError(
                "PRODUCTION requires at least three independent packing replicas"
            )
        for index, item in enumerate(replicas):
            replica = _require_mapping(item, f"replicas[{index}]")
            missing_snapshot_fields = [
                field
                for field in ("input_data", "snapshot_metadata", "snapshot_class")
                if field not in replica
            ]
            if missing_snapshot_fields:
                raise ThermalConfigError(
                    "PRODUCTION replicas must explicitly bind independent snapshot "
                    f"inputs; replicas[{index}] lacks {missing_snapshot_fields}"
                )

    if parent_restart is not None and len(replicas) != 1:
        raise ThermalConfigError("density_parent_restart requires exactly one replica")

    normalized_replicas: list[dict[str, Any]] = []
    for index, item in enumerate(replicas):
        replica = dict(_require_mapping(item, f"replicas[{index}]"))
        effective_class = replica.get("snapshot_class", snapshot_class.value)
        try:
            effective_snapshot_class = SnapshotClass(effective_class)
        except ValueError as exc:
            raise ThermalConfigError(
                f"replicas[{index}].snapshot_class is unsupported"
            ) from exc
        if effective_snapshot_class is SnapshotClass.APG_RAW:
            raise ThermalConfigError(
                f"replicas[{index}] uses APG_RAW, which requires ADEPT eq1/eq2 "
                "preprocessing before MACE thermal MD"
            )
        replica["input_data"] = replica.get("input_data", system["input_data"])
        replica["snapshot_metadata"] = replica.get(
            "snapshot_metadata", system["snapshot_metadata"]
        )
        replica["snapshot_class"] = effective_snapshot_class.value
        for field in ("input_data", "snapshot_metadata"):
            if not isinstance(replica[field], str) or not replica[field]:
                raise ThermalConfigError(
                    f"replicas[{index}].{field} must be a nonempty path"
                )
        normalized_replicas.append(replica)
    if run_class == "PRODUCTION":
        for field in ("input_data", "snapshot_metadata"):
            values = [str(replica[field]) for replica in normalized_replicas]
            if len(set(values)) != len(values):
                raise ThermalConfigError(
                    f"PRODUCTION replicas must use distinct {field} values"
                )
        preparation_classes = {
            str(replica["snapshot_class"]) for replica in normalized_replicas
        }
        if len(preparation_classes) != 1:
            raise ThermalConfigError(
                "PRODUCTION replicas must share one snapshot preparation class"
            )

    state_points: list[StatePointConfig] = []
    state_execution: dict[str, dict[str, Decimal]] = {}
    density_context_value: str
    if mode == "tg":
        tg = _require_mapping(payload.get("tg"), "tg")
        _validate_object_keys(
            tg,
            "tg",
            required={
                "branch",
                "temperatures_k",
                "fit_temperatures_k",
                "density_target_temperature_k",
                "density_context",
            },
            optional={"state_overrides"},
        )
        if tg.get("branch") != "cooling":
            raise ThermalConfigError("tg.branch must be exactly 'cooling'")
        raw_schedule = _require_sequence(tg.get("temperatures_k"), "tg.temperatures_k")
        raw_fit = _require_sequence(
            tg.get("fit_temperatures_k"), "tg.fit_temperatures_k"
        )
        temperatures = [
            _decimal(value, f"tg.temperatures_k[{i}]", positive=True)
            for i, value in enumerate(raw_schedule)
        ]
        fit_temperatures = [
            _decimal(value, f"tg.fit_temperatures_k[{i}]", positive=True)
            for i, value in enumerate(raw_fit)
        ]
        if len(temperatures) < 3 or len(set(temperatures)) != len(temperatures):
            raise ThermalConfigError(
                "tg.temperatures_k must contain at least three unique temperatures"
            )
        if not _strictly_decreasing(temperatures):
            raise ThermalConfigError(
                "tg.temperatures_k must be strictly decreasing to preserve cooling history"
            )
        if len(fit_temperatures) < 3 or len(set(fit_temperatures)) != len(fit_temperatures):
            raise ThermalConfigError(
                "tg.fit_temperatures_k must contain at least three unique explicit fit temperatures"
            )
        missing_fit = [value for value in fit_temperatures if value not in temperatures]
        if missing_fit:
            raise ThermalConfigError(
                f"fit temperatures are absent from the simulated schedule: {missing_fit}"
            )
        raw_density_temperature = tg.get("density_target_temperature_k")
        density_temperature = (
            None
            if raw_density_temperature is None
            else _decimal(
                raw_density_temperature,
                "tg.density_target_temperature_k",
                positive=True,
            )
        )
        density_context = tg.get("density_context")
        density_context_value = str(density_context)
        if density_temperature is None:
            if density_context != "none":
                raise ThermalConfigError(
                    "tg.density_context must be 'none' when no density target is declared"
                )
        else:
            if density_context != "cooling_history":
                raise ThermalConfigError(
                    "a Tg density target must be labeled density_context='cooling_history'"
                )
            if density_temperature not in temperatures:
                raise ThermalConfigError(
                    "tg.density_target_temperature_k must be an exact simulated temperature"
                )
        raw_overrides = _require_sequence(
            tg.get("state_overrides", []), "tg.state_overrides"
        )
        overrides: dict[Decimal, dict[str, Decimal]] = {}
        for override_index, raw_override in enumerate(raw_overrides):
            override = _require_mapping(
                raw_override, f"tg.state_overrides[{override_index}]"
            )
            _validate_object_keys(
                override,
                f"tg.state_overrides[{override_index}]",
                required={"temperature_k"},
                optional={"ramp_ps", "equilibration_ps", "production_ps"},
            )
            override_temperature = _decimal(
                override.get("temperature_k"),
                f"tg.state_overrides[{override_index}].temperature_k",
                positive=True,
            )
            if override_temperature not in temperatures:
                raise ThermalConfigError(
                    "every Tg state override must name an exact simulated temperature"
                )
            if override_temperature in overrides:
                raise ThermalConfigError(
                    f"duplicate Tg state override: {override_temperature} K"
                )
            effective = {
                "ramp_ps": ramp_ps,
                "equilibration_ps": equilibration_ps,
                "production_ps": production_ps,
            }
            for phase in tuple(effective):
                if phase in override:
                    effective[phase] = _decimal(
                        override[phase],
                        f"tg.state_overrides[{override_index}].{phase}",
                        positive=phase != "ramp_ps",
                    )
                    if phase == "ramp_ps" and effective[phase] < 0:
                        raise ThermalConfigError("Tg state ramp_ps must be >= 0")
            for phase, duration in effective.items():
                steps = ps_to_steps(duration, timestep) if duration > 0 else 0
                if steps and steps % sample_steps != 0:
                    raise ThermalConfigError(
                        f"Tg {override_temperature} K {phase} must be an integer "
                        "multiple of npt.sample_interval_ps"
                    )
                if steps and restart_steps > steps:
                    raise ThermalConfigError(
                        f"npt.restart_interval_ps exceeds Tg {override_temperature} K {phase}"
                    )
                if steps and steps % restart_steps != 0:
                    raise ThermalConfigError(
                        f"Tg {override_temperature} K {phase} must be an integer "
                        "multiple of npt.restart_interval_ps"
                    )
            overrides[override_temperature] = effective
        for index, temperature in enumerate(temperatures):
            phase_durations = overrides.get(
                temperature,
                {
                    "ramp_ps": ramp_ps,
                    "equilibration_ps": equilibration_ps,
                    "production_ps": production_ps,
                },
            )
            state_id = f"cool_{index:03d}_{_temperature_label(temperature)}K"
            state_points.append(
                StatePointConfig.constant(
                    state_id=state_id,
                    temperature_k=temperature,
                    pressure_bar=pressure_bar,
                    duration_ps=sum(phase_durations.values(), Decimal("0")),
                    stage_kind=StageKind.PRODUCTION,
                    use_for_tg_fit=temperature in fit_temperatures,
                    use_for_density=temperature == density_temperature,
                )
            )
            state_execution[state_id] = dict(phase_durations)
    else:
        density = _require_mapping(payload.get("density"), "density")
        _validate_object_keys(
            density,
            "density",
            required={"temperature_k", "density_context"},
            optional={"reference_density_g_cm3"},
        )
        if density.get("density_context") != "isothermal_aged":
            raise ThermalConfigError(
                "density.density_context must be exactly 'isothermal_aged'"
            )
        density_context_value = "isothermal_aged"
        temperature = _decimal(
            density.get("temperature_k"), "density.temperature_k", positive=True
        )
        reference_density = density.get("reference_density_g_cm3")
        if reference_density is not None:
            _decimal(
                reference_density,
                "density.reference_density_g_cm3",
                positive=True,
            )
        density_state_id = f"density_{_temperature_label(temperature)}K"
        state_points.append(
            StatePointConfig.constant(
                state_id=density_state_id,
                temperature_k=temperature,
                pressure_bar=pressure_bar,
                duration_ps=ramp_ps + equilibration_ps + production_ps,
                stage_kind=StageKind.PRODUCTION,
                use_for_tg_fit=False,
                use_for_density=True,
            )
        )
        state_execution[density_state_id] = {
            "ramp_ps": ramp_ps,
            "equilibration_ps": equilibration_ps,
            "production_ps": production_ps,
        }

    protocol = ThermalProtocolConfig(
        timestep_ps=float(timestep),
        state_points=tuple(state_points),
        replica_ids=tuple(replica_ids),
    )
    plan = build_thermal_plan(protocol)
    protocol_sha256 = canonical_sha256(
        {
            "schema_version": "thermal-properties-expanded-protocol/v1",
            "base_protocol_sha256": plan.protocol_sha256,
            "ramp_ps": str(ramp_ps),
            "constant_equilibration_ps": str(equilibration_ps),
            "production_ps": str(production_ps),
            "state_execution": {
                state_id: {phase: str(value) for phase, value in phases.items()}
                for state_id, phases in sorted(state_execution.items())
            },
        }
    )
    resolved_steps: list[dict[str, Any]] = []
    equilibration_steps = ps_to_steps(equilibration_ps, timestep)
    production_steps = ps_to_steps(production_ps, timestep)
    initialization_plan: list[dict[str, Any]] = []
    for replica_id in replica_ids:
        initialization_plan.extend(
            (
                {
                    "replica_id": replica_id,
                    "stage_id": f"{replica_id}__initialize",
                    "stage_role": "initialization",
                    "initialize_velocities": True,
                    "predecessor_restart": None,
                    "output_restart": f"restart/{replica_id}/initialize.restart",
                    "seed": replica_seeds[replica_id],
                    "execution_status": "planned",
                    "qc_status": "not_evaluated",
                    "analysis_status": "not_applicable",
                },
                {
                    "replica_id": replica_id,
                    "stage_id": f"{replica_id}__mace_transition_npt",
                    "stage_role": "mace_transition_npt",
                    "initialize_velocities": False,
                    "predecessor_restart": f"restart/{replica_id}/initialize.restart",
                    "output_restart": f"restart/{replica_id}/mace_transition.restart",
                    "seed": replica_seeds[replica_id],
                    "execution_status": "planned",
                    "qc_status": "not_evaluated",
                    "analysis_status": "not_applicable",
                },
            )
        )
    previous_output_by_replica = {
        replica_id: f"restart/{replica_id}/mace_transition.restart"
        for replica_id in replica_ids
    }
    previous_temperature_by_replica = {
        replica_id: float(_decimal(initialize["temperature_k"], "initialize.temperature_k"))
        for replica_id in replica_ids
    }
    for step in plan.steps:
        phase_durations = state_execution[step.state.state_id]
        state_ramp_steps = (
            ps_to_steps(phase_durations["ramp_ps"], timestep)
            if phase_durations["ramp_ps"] > 0
            else 0
        )
        state_equilibration_steps = ps_to_steps(
            phase_durations["equilibration_ps"], timestep
        )
        state_production_steps = ps_to_steps(
            phase_durations["production_ps"], timestep
        )
        record = step.to_dict()
        record["initialize_velocities"] = False
        record["predecessor_restart"] = previous_output_by_replica[step.replica_id]
        record.update(
            {
                "equilibration_steps": state_ramp_steps + state_equilibration_steps,
                "constant_equilibration_steps": state_equilibration_steps,
                "ramp_steps": state_ramp_steps,
                "preproduction_steps": state_ramp_steps + state_equilibration_steps,
                "temperature_ramp_start_k": previous_temperature_by_replica[
                    step.replica_id
                ],
                "temperature_ramp_end_k": step.state.temperature_start_k,
                "production_steps": state_production_steps,
                "phase_durations_ps": {
                    phase: float(duration)
                    for phase, duration in phase_durations.items()
                },
                "seed": replica_seeds[step.replica_id],
                "roles": [
                    role
                    for role, enabled in (
                        ("tg_fit", step.use_for_tg_fit),
                        ("density_target", step.use_for_density),
                    )
                    if enabled
                ],
            }
        )
        resolved_steps.append(record)
        previous_output_by_replica[step.replica_id] = record["output_restart"]
        previous_temperature_by_replica[step.replica_id] = (
            step.state.temperature_end_k
        )

    resolved: dict[str, Any] = {
        "schema_version": RUN_SPEC_SCHEMA,
        "thermal_mode": mode,
        "run_class": run_class,
        "scientific_eligible": scientific_eligible,
        "reuse_eligible": reuse_eligible,
        "quality_tier": payload.get("quality_tier"),
        "density_context": density_context_value,
        "system": system,
        "engine": engine,
        "initialize": initialize,
        "npt": npt,
        "replicas": normalized_replicas,
        "replica_seeds": replica_seeds,
        "protocol": protocol.to_dict(),
        "protocol_sha256": protocol_sha256,
        "plan": {
            "execution_status": plan.execution_status.value,
            "qc_status": plan.qc_status.value,
            "analysis_status": plan.analysis_status.value,
            "initialization": initialization_plan,
            "steps": resolved_steps,
        },
        "source_config_sha256": canonical_sha256(payload),
        "output_root": payload.get("output_root"),
        "scheduler": dict(_require_mapping(payload.get("scheduler"), "scheduler")),
    }
    if reuse_config is not None:
        resolved["reuse"] = reuse_config
    if sampling_continuation is not None:
        expected_transition_ps = 25 if parent_restart is not None else 50
        if (_decimal(initialize["mace_transition_npt_ps"], "transition duration")
                != Decimal(expected_transition_ps)
                or _decimal(npt["production_ps"], "production duration") != Decimal(25)
                or _decimal(payload["density"]["temperature_k"], "density temperature") != Decimal(300)):
            raise ThermalConfigError("sampling_continuation requires initial 50 ps transition (25 ps with parent), 25 ps production at 300 K")
        resolved["sampling_continuation"] = sampling_continuation
    if parent_restart is not None:
        resolved["density_parent_restart"] = parent_restart
        # Intake is provenance, not a newly executed initialization stage.
        transition_plan = initialization_plan[1]
        transition_plan["predecessor_restart"] = "input/parent_restart/restart.final"
        transition_plan["start_step"] = parent_restart["expected_step"]
        transition_plan["parent_restart_sha256"] = parent_restart["restart_sha256"]
        resolved["plan"]["initialization"] = [transition_plan]
    resolved["resolved_spec_sha256"] = resolved_spec_sha256(resolved)
    return resolved


def resolve_thermal_config(path: str | Path) -> dict[str, Any]:
    """Load and expand a high-level JSON configuration without side effects."""

    return expand_thermal_config(_load_json_object(path))


def _guard_density_parent_destination(
    resolved: Mapping[str, Any], output_root: Path, run_id: str
) -> None:
    """Protect the original parent before any output directory is created."""
    parent = resolved.get("density_parent_restart")
    if parent is None:
        return
    parent_root = Path(parent["run_root"]).resolve()
    if run_id == parent["run_id"]:
        raise ThermalSimulationError("parent continuation requires a new run identity")
    for name in (run_id, f"{run_id}.incomplete"):
        destination = (output_root / name).resolve()
        if (
            destination == parent_root
            or destination.is_relative_to(parent_root)
            or parent_root.is_relative_to(destination)
        ):
            raise ThermalSimulationError("new run output overlaps the immutable parent")


def _density_parent_copy_path(input_root: Path, key: str) -> Path:
    filename = "restart.final" if key == "restart" else key
    parent_input = input_root / "parent_restart"
    destination = parent_input / filename
    # Reject redirected intake *before* copying anything into an old parent.
    # Existing new-run resume trees are not necessarily empty.
    if any(path.is_symlink() for path in (input_root, parent_input, destination)):
        raise StageIntegrityError("parent intake destination contains a symlink")
    if not destination.resolve().is_relative_to(input_root.resolve()):
        raise StageIntegrityError("parent intake destination escapes new input root")
    return destination


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_copy_file(source: Path, destination: Path) -> None:
    """Copy one immutable artifact without exposing a partial destination."""

    if not source.is_file() or source.stat().st_size <= 0:
        raise StageIntegrityError(f"source artifact is missing or empty: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.stat().st_size <= 0:
            raise StageIntegrityError(f"destination artifact is empty: {destination}")
        if sha256_file(destination) != sha256_file(source):
            raise StageIntegrityError(
                f"refusing to replace a conflicting finalized artifact: {destination}"
            )
        return
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        if temporary.stat().st_size != source.stat().st_size:
            raise StageIntegrityError(
                f"atomic artifact copy changed size: {source} -> {destination}"
            )
        if sha256_file(temporary) != sha256_file(source):
            raise StageIntegrityError(
                f"atomic artifact copy changed content: {source} -> {destination}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageIntegrityError(f"cannot read required JSON: {path}") from exc
    if not isinstance(value, dict):
        raise StageIntegrityError(f"JSON root must be an object: {path}")
    return value


def _resolve_stage_reference(stage_dir: Path, value: object) -> Path:
    reference = Path(str(value))
    if not reference.is_absolute():
        reference = stage_dir / reference
    return reference.resolve()


def artifact_record(path: str | Path, *, relative_to: str | Path) -> dict[str, Any]:
    source = Path(path)
    root = Path(relative_to).resolve()
    try:
        relative = source.resolve().relative_to(root)
    except ValueError as exc:
        raise StageIntegrityError(f"artifact escapes run root: {source}") from exc
    if not source.is_file():
        raise StageIntegrityError(f"artifact is missing: {source}")
    return {
        "path": str(relative),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def verify_artifact_record(record: Mapping[str, Any], *, relative_to: str | Path) -> Path:
    root = Path(relative_to).resolve()
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise StageIntegrityError("artifact record has no path")
    candidate = (root / raw_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise StageIntegrityError(f"artifact path escapes root: {raw_path}") from exc
    if not candidate.is_file():
        raise StageIntegrityError(f"artifact is missing: {candidate}")
    if candidate.stat().st_size != int(record.get("bytes", -1)):
        raise StageIntegrityError(f"artifact size mismatch: {candidate}")
    if sha256_file(candidate) != record.get("sha256"):
        raise StageIntegrityError(f"artifact hash mismatch: {candidate}")
    return candidate


def record_checkpoint(
    restart_path: str | Path,
    *,
    stage_dir: str | Path,
    completed_equilibration_steps: int,
    completed_production_steps: int,
) -> Path:
    """Create a hash sidecar that makes a periodic restart eligible for resume."""

    restart = Path(restart_path)
    stage = Path(stage_dir)
    if not restart.is_file() or restart.stat().st_size <= 0:
        raise StageIntegrityError(f"checkpoint is missing or empty: {restart}")
    for name, value in (
        ("completed_equilibration_steps", completed_equilibration_steps),
        ("completed_production_steps", completed_production_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise StageIntegrityError(f"{name} must be a nonnegative integer")
    stage_spec_path = stage / "stage_spec.json"
    if not stage_spec_path.is_file():
        raise StageIntegrityError(
            f"cannot bind checkpoint without stage_spec.json: {stage}"
        )
    stage_spec = _read_json(stage_spec_path)
    initialization_only = stage_spec.get("stage_role") == "initialization"
    has_equilibration_phase = int(stage_spec.get("equilibration_steps", 0)) > 0
    predecessor_path = stage_spec.get("predecessor_restart")
    predecessor_sha256: str | None = None
    if predecessor_path:
        predecessor = _resolve_stage_reference(stage, predecessor_path)
        if not predecessor.is_file():
            raise StageIntegrityError(
                f"checkpoint predecessor is missing: {predecessor}"
            )
        predecessor_sha256 = sha256_file(predecessor)
    status_path = stage / "stage_status.json"
    status = _read_json(status_path) if status_path.is_file() else {}
    attempt_index = status.get("segment_index")
    if isinstance(attempt_index, bool) or not isinstance(attempt_index, int) or attempt_index < 0:
        raise StageIntegrityError(
            f"cannot bind checkpoint without a valid attempt index: {restart}"
        )
    attempt_spec_path = stage / "attempts" / f"attempt_{attempt_index:04d}.json"
    if not attempt_spec_path.is_file():
        raise StageIntegrityError(
            f"cannot bind checkpoint without immutable attempt spec: {restart}"
        )
    record = {
        "schema_version": CHECKPOINT_SCHEMA,
        "stage_spec_sha256": canonical_sha256(stage_spec),
        "predecessor_restart_sha256": predecessor_sha256,
        "attempt_index": attempt_index,
        "attempt_spec_sha256": sha256_file(attempt_spec_path),
        "completed_equilibration_steps": completed_equilibration_steps,
        "completed_production_steps": completed_production_steps,
        "artifact": artifact_record(restart, relative_to=stage),
    }
    if (
        completed_production_steps > 0
        and not initialization_only
        and has_equilibration_phase
    ):
        equilibration_restart = stage / "restart.equilibration"
        if not equilibration_restart.is_file():
            raise StageIntegrityError(
                "production checkpoint lacks restart.equilibration"
            )
        record["equilibration_final_restart"] = artifact_record(
            equilibration_restart, relative_to=stage
        )
    sidecar = restart.with_suffix(restart.suffix + ".json")
    _atomic_write_json(sidecar, record)
    return sidecar


def verified_checkpoints(
    stage_dir: str | Path,
    *,
    planned_equilibration_steps: int,
    planned_production_steps: int,
) -> tuple[CheckpointRecord, ...]:
    """Return every valid checkpoint; malformed sidecars fail closed."""

    stage = Path(stage_dir)
    checkpoint_dir = stage / "checkpoints"
    if not checkpoint_dir.exists():
        return ()
    candidates: list[CheckpointRecord] = []
    stage_spec_path = stage / "stage_spec.json"
    if not stage_spec_path.is_file():
        raise StageIntegrityError(f"stage spec is missing: {stage_spec_path}")
    stage_spec = _read_json(stage_spec_path)
    initialization_only = stage_spec.get("stage_role") == "initialization"
    has_equilibration_phase = int(stage_spec.get("equilibration_steps", 0)) > 0
    expected_spec_sha256 = canonical_sha256(stage_spec)
    predecessor_path = stage_spec.get("predecessor_restart")
    expected_predecessor_sha256: str | None = None
    if predecessor_path:
        predecessor = _resolve_stage_reference(stage, predecessor_path)
        if not predecessor.is_file():
            raise StageIntegrityError(
                f"stage predecessor is missing during resume: {predecessor}"
            )
        expected_predecessor_sha256 = sha256_file(predecessor)
    for sidecar in sorted(checkpoint_dir.glob("*.restart.json")):
        payload = _read_json(sidecar)
        if payload.get("schema_version") != CHECKPOINT_SCHEMA:
            raise StageIntegrityError(f"unsupported checkpoint schema: {sidecar}")
        if payload.get("stage_spec_sha256") != expected_spec_sha256:
            raise StageIntegrityError(f"checkpoint belongs to another stage spec: {sidecar}")
        if payload.get("predecessor_restart_sha256") != expected_predecessor_sha256:
            raise StageIntegrityError(
                f"checkpoint predecessor lineage mismatch: {sidecar}"
            )
        artifact = payload.get("artifact")
        if not isinstance(artifact, Mapping):
            raise StageIntegrityError(f"checkpoint artifact missing: {sidecar}")
        restart = verify_artifact_record(artifact, relative_to=stage)
        equil = payload.get("completed_equilibration_steps")
        prod = payload.get("completed_production_steps")
        attempt_index = payload.get("attempt_index")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (equil, prod)):
            raise StageIntegrityError(f"invalid checkpoint progress: {sidecar}")
        if (
            isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or attempt_index < 0
        ):
            raise StageIntegrityError(f"invalid checkpoint attempt index: {sidecar}")
        attempt_spec_path = (
            stage / "attempts" / f"attempt_{attempt_index:04d}.json"
        )
        if (
            not attempt_spec_path.is_file()
            or sha256_file(attempt_spec_path) != payload.get("attempt_spec_sha256")
        ):
            raise StageIntegrityError(
                f"checkpoint attempt lineage is invalid: {sidecar}"
            )
        if equil > planned_equilibration_steps or prod > planned_production_steps:
            raise StageIntegrityError(f"checkpoint exceeds planned work: {sidecar}")
        if prod > 0 and equil != planned_equilibration_steps:
            raise StageIntegrityError(
                f"production checkpoint lacks completed equilibration: {sidecar}"
            )
        if prod > 0 and not initialization_only and has_equilibration_phase:
            equilibration_restart = payload.get("equilibration_final_restart")
            if not isinstance(equilibration_restart, Mapping):
                raise StageIntegrityError(
                    f"production checkpoint lacks equilibration restart: {sidecar}"
                )
            verify_artifact_record(equilibration_restart, relative_to=stage)
        candidates.append(
            CheckpointRecord(
                path=restart,
                sha256=str(artifact["sha256"]),
                size_bytes=int(artifact["bytes"]),
                completed_equilibration_steps=equil,
                completed_production_steps=prod,
                attempt_index=attempt_index,
                sidecar_path=sidecar.resolve(),
                sidecar_sha256=sha256_file(sidecar),
            )
        )
    if not candidates:
        return ()
    candidates.sort(key=lambda item: (item.completed_steps, str(item.path)))
    return tuple(candidates)


def latest_verified_checkpoint(
    stage_dir: str | Path,
    *,
    planned_equilibration_steps: int,
    planned_production_steps: int,
) -> CheckpointRecord | None:
    candidates = verified_checkpoints(
        stage_dir,
        planned_equilibration_steps=planned_equilibration_steps,
        planned_production_steps=planned_production_steps,
    )
    return candidates[-1] if candidates else None


_RAW_CHECKPOINT_RE = re.compile(
    r"^(initialize|equilibration|production)\.checkpoint\.(\d+)\.restart$"
)


def promote_stage_boundary_restarts(
    stage_dir: str | Path,
    *,
    require_complete: bool,
) -> dict[str, Path]:
    """Promote exact periodic checkpoints to stable stage boundary names.

    The Kokkos/Intel-MPI path can fail when LAMMPS performs an additional
    standalone ``write_restart`` after a successful run.  Periodic restart
    files are already durable at every configured phase endpoint, so Python
    verifies the exact timestep and atomically gives those files the stable
    names consumed by downstream stages.
    """

    stage = Path(stage_dir)
    spec = _read_json(stage / "stage_spec.json")
    start_step = int(spec.get("start_step", 0))
    planned_equil = int(spec["equilibration_steps"])
    planned_prod = int(spec["production_steps"])
    initialization_only = spec.get("stage_role") == "initialization"
    boundaries: list[tuple[str, Path, Path]] = []
    if initialization_only:
        boundaries.append(
            (
                "final",
                stage
                / "checkpoints"
                / f"initialize.checkpoint.{start_step + planned_prod}.restart",
                stage / "restart.final",
            )
        )
    else:
        if planned_equil > 0:
            boundaries.append(
                (
                    "equilibration",
                    stage
                    / "checkpoints"
                    / f"equilibration.checkpoint.{start_step + planned_equil}.restart",
                    stage / "restart.equilibration",
                )
            )
        boundaries.append(
            (
                "final",
                stage
                / "checkpoints"
                / (
                    "production.checkpoint."
                    f"{start_step + planned_equil + planned_prod}.restart"
                ),
                stage / "restart.final",
            )
        )

    promoted: dict[str, Path] = {}
    for name, checkpoint, destination in boundaries:
        if checkpoint.is_file() and checkpoint.stat().st_size > 0:
            _atomic_copy_file(checkpoint, destination)
            promoted[name] = destination
            continue
        if destination.is_file() and destination.stat().st_size > 0:
            promoted[name] = destination
            continue
        if require_complete:
            raise StageIntegrityError(
                f"successful process lacks exact {name} checkpoint: {checkpoint}"
            )
    return promoted


def register_discovered_checkpoints(stage_dir: str | Path) -> list[Path]:
    """Hash newly written LAMMPS restart files into checkpoint sidecars.

    LAMMPS substitutes the absolute timestep for ``*`` in the configured
    filename.  ``start_step`` in the immutable stage spec converts that value
    into per-stage equilibration/production progress.  A restart outside the
    declared progress interval is rejected rather than guessed.
    """

    stage = Path(stage_dir)
    spec = _read_json(stage / "stage_spec.json")
    start_step = int(spec.get("start_step", 0))
    planned_equil = int(spec["equilibration_steps"])
    planned_prod = int(spec["production_steps"])
    created: list[Path] = []
    for restart in sorted((stage / "checkpoints").glob("*.restart")):
        sidecar = restart.with_suffix(restart.suffix + ".json")
        if sidecar.exists():
            continue
        match = _RAW_CHECKPOINT_RE.match(restart.name)
        if match is None:
            raise StageIntegrityError(
                f"unrecognized checkpoint filename cannot be resumed: {restart}"
            )
        phase, raw_step = match.groups()
        absolute_step = int(raw_step)
        progress = absolute_step - start_step
        if phase == "initialize":
            equil_completed = 0
            prod_completed = progress
        elif phase == "equilibration":
            equil_completed = progress
            prod_completed = 0
        else:
            equil_completed = planned_equil
            prod_completed = progress - planned_equil
        if not (0 <= equil_completed <= planned_equil) or not (
            0 <= prod_completed <= planned_prod
        ):
            raise StageIntegrityError(
                f"checkpoint timestep is outside planned stage: {restart}"
            )
        created.append(
            record_checkpoint(
                restart,
                stage_dir=stage,
                completed_equilibration_steps=equil_completed,
                completed_production_steps=prod_completed,
            )
        )
    return created


def merge_csv_segments(
    segment_paths: Iterable[str | Path],
    destination: str | Path,
    *,
    step_column: str = "step",
    minimum_step_exclusive_by_segment: Mapping[str | Path, int | None] | None = None,
    maximum_step_by_segment: Mapping[str | Path, int | None] | None = None,
    allowed_boundary_duplicate_steps_by_segment: Mapping[
        str | Path, Iterable[int]
    ]
    | None = None,
    expected_step_stride: int | None = None,
) -> dict[str, Any]:
    """Validate and atomically merge sample segments.

    Duplicate steps are rejected by default.  A caller may authenticate an
    expected LAMMPS run boundary within one raw segment; only one consecutive
    duplicate at that exact step is then normalized by retaining the later row.
    """

    segments = [Path(item) for item in segment_paths]
    if not segments:
        raise StageIntegrityError("at least one sample segment is required")
    header: list[str] | None = None
    rows_by_step: dict[int, dict[str, str]] = {}
    normalized_cutoffs = {
        Path(path).resolve(): cutoff
        for path, cutoff in (maximum_step_by_segment or {}).items()
    }
    normalized_minimums = {
        Path(path).resolve(): minimum
        for path, minimum in (minimum_step_exclusive_by_segment or {}).items()
    }
    normalized_boundary_duplicates: dict[Path, set[int]] = {}
    for path, steps in (allowed_boundary_duplicate_steps_by_segment or {}).items():
        normalized_steps: set[int] = set()
        for step in steps:
            if isinstance(step, bool) or not isinstance(step, int):
                raise StageIntegrityError(
                    "allowed boundary duplicate steps must be integers"
                )
            normalized_steps.add(step)
        normalized_boundary_duplicates[Path(path).resolve()] = normalized_steps
    discarded_after_checkpoint: dict[str, int] = {}
    normalized_boundary_duplicate_rows: dict[str, list[int]] = {}
    row_source_by_step: dict[int, Path] = {}
    normalized_boundary_keys: set[tuple[Path, int]] = set()
    for segment in segments:
        if not segment.is_file():
            raise StageIntegrityError(f"sample segment is missing: {segment}")
        previous_included_step: int | None = None
        with segment.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            current_header = reader.fieldnames
            if not current_header or step_column not in current_header:
                raise StageIntegrityError(
                    f"sample segment lacks {step_column!r}: {segment}"
                )
            if header is None:
                header = list(current_header)
            elif current_header != header:
                raise StageIntegrityError(f"sample headers differ: {segment}")
            for row_index, row in enumerate(reader, start=2):
                try:
                    numeric = Decimal(str(row[step_column]))
                    if numeric != numeric.to_integral_value():
                        raise ValueError
                    step = int(numeric)
                except (InvalidOperation, TypeError, ValueError) as exc:
                    raise StageIntegrityError(
                        f"invalid step at {segment}:{row_index}"
                    ) from exc
                resolved_segment = segment.resolve()
                if resolved_segment in normalized_minimums:
                    minimum = normalized_minimums[resolved_segment]
                    if minimum is None or step <= minimum:
                        key = str(segment)
                        discarded_after_checkpoint[key] = (
                            discarded_after_checkpoint.get(key, 0) + 1
                        )
                        continue
                if resolved_segment in normalized_cutoffs:
                    cutoff = normalized_cutoffs[resolved_segment]
                    if cutoff is None or step > cutoff:
                        key = str(segment)
                        discarded_after_checkpoint[key] = (
                            discarded_after_checkpoint.get(key, 0) + 1
                        )
                        continue
                if step in rows_by_step:
                    boundary_key = (resolved_segment, step)
                    if (
                        row_source_by_step[step] == resolved_segment
                        and step == previous_included_step
                        and step
                        in normalized_boundary_duplicates.get(
                            resolved_segment, set()
                        )
                        and boundary_key not in normalized_boundary_keys
                    ):
                        rows_by_step[step] = dict(row)
                        normalized_boundary_keys.add(boundary_key)
                        normalized_boundary_duplicate_rows.setdefault(
                            str(segment), []
                        ).append(step)
                        previous_included_step = step
                        continue
                    raise DuplicateSampleStepError(
                        f"duplicate sample step {step} while merging {segment}"
                    )
                rows_by_step[step] = dict(row)
                row_source_by_step[step] = resolved_segment
                previous_included_step = step
    if header is None:
        raise StageIntegrityError("sample segments have no header")
    ordered_steps = sorted(rows_by_step)
    if expected_step_stride is not None:
        if (
            isinstance(expected_step_stride, bool)
            or not isinstance(expected_step_stride, int)
            or expected_step_stride <= 0
        ):
            raise StageIntegrityError("expected_step_stride must be a positive integer")
        gaps = [
            (left, right)
            for left, right in zip(ordered_steps, ordered_steps[1:])
            if right - left != expected_step_stride
        ]
        if gaps:
            raise StageIntegrityError(
                f"canonical sample cadence has gaps or overlaps: {gaps[:5]}"
            )
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(
        f".{destination_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            for step in ordered_steps:
                writer.writerow(rows_by_step[step])
        os.replace(temporary, destination_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "row_count": len(ordered_steps),
        "first_step": ordered_steps[0] if ordered_steps else None,
        "last_step": ordered_steps[-1] if ordered_steps else None,
        "sha256": sha256_file(destination_path),
        "discarded_rows_after_resume_checkpoint": discarded_after_checkpoint,
        "normalized_boundary_duplicate_steps": (
            normalized_boundary_duplicate_rows
        ),
    }


def _attempt_lineage_ranges(
    stage: Path,
    *,
    current_attempt_index: int,
    selected_checkpoint: CheckpointRecord | None,
    planned_equilibration_steps: int,
    planned_production_steps: int,
    start_step: int,
) -> tuple[dict[Path, int | None], dict[Path, int | None], dict[Path, int | None], dict[Path, int | None], list[int]]:
    """Return non-overlapping sample intervals along the selected resume branch."""

    attempt_specs: dict[int, dict[str, Any]] = {}
    for path in sorted((stage / "attempts").glob("attempt_*.json")):
        match = re.fullmatch(r"attempt_(\d{4})\.json", path.name)
        if match is None:
            raise StageIntegrityError(f"invalid attempt-spec filename: {path}")
        index = int(match.group(1))
        payload = _read_json(path)
        if payload.get("attempt_index") != index:
            raise StageIntegrityError(f"attempt identity mismatch: {path}")
        attempt_specs[index] = payload
    if current_attempt_index not in attempt_specs:
        raise StageIntegrityError("current attempt spec is missing")

    checkpoints = verified_checkpoints(
        stage,
        planned_equilibration_steps=planned_equilibration_steps,
        planned_production_steps=planned_production_steps,
    )
    checkpoint_by_sidecar_hash = {
        item.sidecar_sha256: item for item in checkpoints
    }
    lineage_reversed: list[int] = [current_attempt_index]
    seen = {current_attempt_index}
    current_spec = attempt_specs[current_attempt_index]
    parent = current_spec.get("parent_checkpoint")
    if selected_checkpoint is None:
        if parent is not None:
            raise StageIntegrityError("attempt unexpectedly declares a parent checkpoint")
    else:
        if not isinstance(parent, Mapping) or parent.get(
            "sidecar_sha256"
        ) != selected_checkpoint.sidecar_sha256:
            raise StageIntegrityError(
                "current attempt is not bound to the selected resume checkpoint"
            )
    while parent is not None:
        if not isinstance(parent, Mapping):
            raise StageIntegrityError("attempt parent checkpoint is malformed")
        parent_hash = parent.get("sidecar_sha256")
        checkpoint = checkpoint_by_sidecar_hash.get(parent_hash)
        if checkpoint is None:
            raise StageIntegrityError(
                "attempt references an unverified checkpoint sidecar"
            )
        producer = checkpoint.attempt_index
        if producer in seen or producer not in attempt_specs:
            raise StageIntegrityError("attempt checkpoint lineage is cyclic or incomplete")
        seen.add(producer)
        lineage_reversed.append(producer)
        parent = attempt_specs[producer].get("parent_checkpoint")
    lineage = list(reversed(lineage_reversed))

    equil_minimums: dict[Path, int | None] = {}
    equil_maximums: dict[Path, int | None] = {}
    prod_minimums: dict[Path, int | None] = {}
    prod_maximums: dict[Path, int | None] = {}
    all_equil = sorted((stage / "segments").glob("equilibration.segment_*.csv"))
    all_prod = sorted((stage / "segments").glob("production.segment_*.csv"))
    for path in all_equil:
        equil_minimums[path] = None
        equil_maximums[path] = None
    for path in all_prod:
        prod_minimums[path] = None
        prod_maximums[path] = None

    for position, index in enumerate(lineage):
        spec = attempt_specs[index]
        parent_record = spec.get("parent_checkpoint")
        if parent_record is None:
            parent_equil = 0
            parent_prod = 0
        else:
            parent_equil = int(parent_record["completed_equilibration_steps"])
            parent_prod = int(parent_record["completed_production_steps"])
        if position + 1 < len(lineage):
            child_parent = attempt_specs[lineage[position + 1]].get(
                "parent_checkpoint"
            )
            if not isinstance(child_parent, Mapping):
                raise StageIntegrityError("resume lineage child lacks its parent")
            end_equil = int(child_parent["completed_equilibration_steps"])
            end_prod = int(child_parent["completed_production_steps"])
        else:
            end_equil = planned_equilibration_steps
            end_prod = planned_production_steps
        equil_path = _resolve_stage_reference(stage, spec["equilibration_segment"])
        prod_path = _resolve_stage_reference(stage, spec["production_segment"])
        if equil_path.is_file() and end_equil > parent_equil:
            equil_minimums[equil_path] = start_step + parent_equil
            equil_maximums[equil_path] = start_step + end_equil
        if prod_path.is_file() and end_prod > parent_prod:
            prod_minimums[prod_path] = (
                start_step + planned_equilibration_steps + parent_prod
            )
            prod_maximums[prod_path] = (
                start_step + planned_equilibration_steps + end_prod
            )
    return (
        equil_minimums,
        equil_maximums,
        prod_minimums,
        prod_maximums,
        lineage,
    )


def _runtime_process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Existence without permission to signal is not confirmed cleanup.
        return True
    return True


def _terminate_runtime_process_group(process: subprocess.Popen[Any]) -> bool:
    """Bound cleanup by group lifetime, even when its launcher has exited."""

    for termination_signal, grace_seconds in (
        (signal.SIGTERM, _FATAL_RUNTIME_TERMINATION_GRACE_SECONDS),
        (signal.SIGKILL, _FATAL_RUNTIME_KILL_GRACE_SECONDS),
    ):
        process.poll()  # Reap our direct child so it cannot keep the group alive.
        if not _runtime_process_group_exists(process.pid):
            return True
        try:
            os.killpg(process.pid, termination_signal)
        except ProcessLookupError:
            process.poll()
            return True
        except PermissionError:
            return False
        deadline = time.monotonic() + grace_seconds
        while True:
            process.poll()
            if not _runtime_process_group_exists(process.pid):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(_FATAL_RUNTIME_POLL_SECONDS, remaining))
    # A zombie awaiting its new parent or an uninterruptible group member
    # must not be represented as proven group cleanup.
    process.poll()
    return not _runtime_process_group_exists(process.pid)


def _default_process_runner(invocation: StageInvocation) -> int:
    environment = os.environ.copy()
    environment.update(invocation.environment)
    invocation.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    log_paths = (invocation.stdout_path, invocation.stderr_path)
    log_offsets = {
        path: path.stat().st_size if path.is_file() else 0 for path in log_paths
    }
    log_tails = {path: b"" for path in log_paths}
    overlap_bytes = max(len(pattern) for pattern in _FATAL_RUNTIME_STDERR_PATTERNS) - 1

    def find_fatal_output() -> tuple[bytes, Path] | None:
        for path in log_paths:
            with path.open("rb") as reader:
                reader.seek(log_offsets[path])
                while appended := reader.read(65536):
                    log_offsets[path] = reader.tell()
                    window = log_tails[path] + appended
                    # Scan the complete new block before retaining only the
                    # overlap needed for a signature split across reads.
                    for pattern in _FATAL_RUNTIME_STDERR_PATTERNS:
                        if pattern in window:
                            return pattern, path
                    log_tails[path] = window[-overlap_bytes:]
        return None

    with invocation.stdout_path.open("a", encoding="utf-8") as stdout_handle, invocation.stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            list(invocation.command),
            cwd=invocation.cwd,
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        try:
            while True:
                # Poll before reading: when the child has exited, this scan
                # also includes its final writes before returning its status.
                return_code = process.poll()
                fatal_output = find_fatal_output()
                if fatal_output is not None:
                    fatal_pattern, fatal_path = fatal_output
                    group_terminated = _terminate_runtime_process_group(process)
                    stderr_handle.write(
                        "\nTHERMAL_FATAL_RUNTIME_DETECTED="
                        f"{fatal_pattern.decode('ascii')}; "
                        f"source={fatal_path.name}; "
                        f"process_group_id={process.pid}; "
                        f"terminated_process_group={'yes' if group_terminated else 'no'}; "
                        "process_group_cleanup="
                        f"{'CONFIRMED' if group_terminated else 'UNCONFIRMED'}\n"
                    )
                    stderr_handle.flush()
                    os.fsync(stderr_handle.fileno())
                    return _FATAL_RUNTIME_RETURN_CODE
                if return_code is not None:
                    return int(return_code)
                time.sleep(_FATAL_RUNTIME_POLL_SECONDS)
        except BaseException:
            _terminate_runtime_process_group(process)
            raise


def _return_code(result: int | subprocess.CompletedProcess[Any]) -> int:
    if isinstance(result, subprocess.CompletedProcess):
        return int(result.returncode)
    if isinstance(result, bool) or not isinstance(result, int):
        raise ThermalSimulationError("process runner must return int or CompletedProcess")
    return result


def _stage_is_reusable(stage_dir: Path, expected_spec_sha256: str) -> bool:
    status_path = stage_dir / "stage_status.json"
    manifest_path = stage_dir / "stage_manifest.json"
    if not status_path.is_file() or not manifest_path.is_file():
        return False
    status = _read_json(status_path)
    if status.get("execution_status") != "COMPLETE":
        return False
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != STAGE_MANIFEST_SCHEMA:
        raise StageIntegrityError(f"unsupported stage manifest: {manifest_path}")
    if manifest.get("stage_spec_sha256") != expected_spec_sha256:
        raise StageIntegrityError(f"completed stage spec does not match: {stage_dir}")
    manifest_sha256 = sha256_file(manifest_path)
    if status.get("stage_manifest_sha256") != manifest_sha256:
        raise StageIntegrityError(
            f"stage status does not authenticate its manifest: {stage_dir}"
        )
    for field in ("execution_status", "qc_status", "analysis_status"):
        if status.get(field) != manifest.get(field):
            raise StageIntegrityError(
                f"stage status/manifest disagree on {field}: {stage_dir}"
            )
    if manifest.get("execution_status") != "COMPLETE":
        raise StageIntegrityError(f"stage manifest is not complete: {stage_dir}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise StageIntegrityError(f"completed stage lacks artifacts: {stage_dir}")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise StageIntegrityError(f"malformed artifact record: {stage_dir}")
        verify_artifact_record(artifact, relative_to=stage_dir)
    return True


def _persist_incomplete_on_error(function):
    """Ensure post-MD validation failures cannot leave a RUNNING status."""

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException as exc:
            raw_stage = kwargs.get("stage_dir")
            if raw_stage is not None:
                stage = Path(raw_stage)
                status_path = stage / "stage_status.json"
                try:
                    status = _read_json(status_path) if status_path.is_file() else {}
                    if status.get("execution_status") == "RUNNING":
                        status.update(
                            {
                                "execution_status": "INCOMPLETE",
                                "qc_status": "NOT_EVALUATED",
                                "analysis_status": "NOT_REQUESTED",
                                "failure_phase": "process_or_artifact_finalization",
                                "failure_type": type(exc).__name__,
                                "failure_message": str(exc),
                            }
                        )
                        _atomic_write_json(status_path, status)
                except Exception:
                    # Preserve the original execution/finalization exception.
                    pass
            raise

    return wrapped


@_persist_incomplete_on_error
def execute_npt_stage(
    *,
    stage_dir: str | Path,
    stage_spec: Mapping[str, Any],
    command: Sequence[str],
    environment: Mapping[str, str] | None = None,
    process_runner: ProcessRunner | None = None,
    qc_evaluator: Callable[[Path, Path], str | Mapping[str, Any]] | None = None,
) -> StageExecutionResult:
    """Execute or resume exactly one NPT state point.

    The injected runner must write the two invocation segment paths, the final
    restart, and ``log.lammps``.  On interruption it may write hash-sidecarred
    checkpoints with :func:`record_checkpoint`.  Existing completed work is
    skipped only after every manifest artifact verifies.
    """

    stage = Path(stage_dir)
    stage.mkdir(parents=True, exist_ok=True)
    for child in ("attempts", "checkpoints", "segments", "samples"):
        (stage / child).mkdir(exist_ok=True)
    required = {
        "stage_id",
        "replica_id",
        "equilibration_steps",
        "production_steps",
        "initialize_velocities",
    }
    missing = sorted(required.difference(stage_spec))
    if missing:
        raise ThermalSimulationError(f"stage spec missing required fields: {missing}")
    planned_equil = int(stage_spec["equilibration_steps"])
    planned_prod = int(stage_spec["production_steps"])
    initialization_only = stage_spec.get("stage_role") == "initialization"
    if planned_equil < 0 or planned_prod <= 0:
        raise ThermalSimulationError("planned stage steps are invalid")
    normalized_spec = dict(stage_spec)
    normalized_spec["schema_version"] = STAGE_SPEC_SCHEMA
    spec_sha256 = canonical_sha256(normalized_spec)
    spec_path = stage / "stage_spec.json"
    if spec_path.exists():
        existing = _read_json(spec_path)
        if canonical_sha256(existing) != spec_sha256:
            raise StageIntegrityError(f"stage spec changed during resume: {stage}")
    else:
        _atomic_write_json(spec_path, normalized_spec)

    if _stage_is_reusable(stage, spec_sha256):
        manifest = _read_json(stage / "stage_manifest.json")
        return StageExecutionResult(
            stage_id=str(stage_spec["stage_id"]),
            stage_dir=stage,
            execution_status="COMPLETE",
            qc_status=str(manifest.get("qc_status", "NOT_EVALUATED")),
            analysis_status=str(manifest.get("analysis_status", "NOT_REQUESTED")),
            skipped=True,
            resumed_from=None,
            manifest_path=stage / "stage_manifest.json",
        )

    register_discovered_checkpoints(stage)
    checkpoint = latest_verified_checkpoint(
        stage,
        planned_equilibration_steps=planned_equil,
        planned_production_steps=planned_prod,
    )
    completed_equil = checkpoint.completed_equilibration_steps if checkpoint else 0
    completed_prod = checkpoint.completed_production_steps if checkpoint else 0
    resume_reference = (
        os.path.relpath(checkpoint.path, start=stage) if checkpoint else None
    )
    remaining_equil = planned_equil - completed_equil
    remaining_prod = planned_prod - completed_prod
    planned_ramp = int(stage_spec.get("ramp_steps", 0))
    planned_constant_equil = int(
        stage_spec.get("constant_equilibration_steps", planned_equil - planned_ramp)
    )
    if initialization_only:
        planned_ramp = 0
        planned_constant_equil = 0
    elif (
        planned_ramp < 0
        or planned_constant_equil < 0
        or planned_ramp + planned_constant_equil != planned_equil
    ):
        raise ThermalSimulationError(
            "ramp_steps plus constant_equilibration_steps must equal "
            "equilibration_steps"
        )
    completed_ramp = min(completed_equil, planned_ramp)
    remaining_ramp = planned_ramp - completed_ramp
    completed_constant_equil = max(0, completed_equil - planned_ramp)
    remaining_constant_equil = (
        planned_constant_equil - completed_constant_equil
    )
    ramp_start_temperature = stage_spec.get(
        "temperature_ramp_start_k",
        stage_spec.get("state", {}).get("temperature_start_k")
        if isinstance(stage_spec.get("state"), Mapping)
        else None,
    )
    ramp_end_temperature = stage_spec.get(
        "temperature_ramp_end_k",
        stage_spec.get("state", {}).get("temperature_start_k")
        if isinstance(stage_spec.get("state"), Mapping)
        else None,
    )
    if not initialization_only:
        if ramp_start_temperature is None or ramp_end_temperature is None:
            if planned_ramp:
                raise ThermalSimulationError(
                    "ramped NPT state lacks explicit ramp start/end temperatures"
                )
            # Legacy low-level zero-ramp test stages do not execute the real
            # LAMMPS input and may omit physical setpoints.  Zero is safe here
            # because the ramp command is skipped; campaign specs always pin
            # the actual target values.
            ramp_start_decimal = Decimal("0")
            ramp_end_decimal = Decimal("0")
        else:
            ramp_start_decimal = _decimal(
                ramp_start_temperature,
                "temperature_ramp_start_k",
                positive=True,
            )
            ramp_end_decimal = _decimal(
                ramp_end_temperature, "temperature_ramp_end_k", positive=True
            )
        if planned_ramp and completed_ramp:
            fraction = Decimal(completed_ramp) / Decimal(planned_ramp)
            ramp_start_decimal = ramp_start_decimal + (
                ramp_end_decimal - ramp_start_decimal
            ) * fraction
        if remaining_ramp == 0:
            ramp_start_decimal = ramp_end_decimal
    existing_indices: list[int] = []
    pattern = re.compile(r"(?:equilibration|production)\.segment_(\d{4})\.csv$")
    for path in (stage / "segments").glob("*.csv"):
        match = pattern.search(path.name)
        if match:
            existing_indices.append(int(match.group(1)))
    for path in (stage / "attempts").glob("attempt_*.json"):
        match = re.fullmatch(r"attempt_(\d{4})\.json", path.name)
        if match:
            existing_indices.append(int(match.group(1)))
    if initialization_only and existing_indices and checkpoint is None:
        raise StageIntegrityError(
            "initialization was interrupted without a verified checkpoint; "
            "refusing to recreate velocities or repeat initialization"
        )
    segment_index = max(existing_indices, default=-1) + 1
    equil_segment = stage / "segments" / f"equilibration.segment_{segment_index:04d}.csv"
    prod_segment = stage / "segments" / f"production.segment_{segment_index:04d}.csv"
    final_restart = stage / "restart.final"
    stdout_path = stage / "stdout.log"
    stderr_path = stage / "stderr.log"
    log_path = stage / "log.lammps"
    attempt_spec = {
        "schema_version": "thermal-properties-stage-attempt/v1",
        "attempt_index": segment_index,
        "parent_checkpoint": (
            {
                "sidecar_path": os.path.relpath(checkpoint.sidecar_path, start=stage),
                "sidecar_sha256": checkpoint.sidecar_sha256,
                "restart_sha256": checkpoint.sha256,
                "completed_equilibration_steps": completed_equil,
                "completed_production_steps": completed_prod,
            }
            if checkpoint
            else None
        ),
        "equilibration_segment": os.path.relpath(equil_segment, start=stage),
        "production_segment": os.path.relpath(prod_segment, start=stage),
        "allowed_equilibration_boundary_duplicate_steps": (
            [int(stage_spec.get("start_step", 0)) + planned_ramp]
            if remaining_ramp > 0 and remaining_constant_equil > 0
            else []
        ),
    }
    _atomic_write_json(
        stage / "attempts" / f"attempt_{segment_index:04d}.json",
        attempt_spec,
    )
    resolved_environment = {str(key): str(value) for key, value in (environment or {}).items()}
    resolved_environment.update(
        {
            "THERMAL_STAGE_ID": str(stage_spec["stage_id"]),
            "THERMAL_REPLICA_ID": str(stage_spec["replica_id"]),
            "THERMAL_RAMP_STEPS": str(remaining_ramp),
            "THERMAL_EQUIL_STEPS": str(remaining_constant_equil),
            "THERMAL_PROD_STEPS": str(remaining_prod),
            "THERMAL_EQUIL_CSV": str(equil_segment.resolve()),
            "THERMAL_PROD_CSV": str(prod_segment.resolve()),
            "THERMAL_PROD_FINAL_RESTART": str(final_restart.resolve()),
            "THERMAL_LAMMPS_LOG": str(log_path.resolve()),
            "THERMAL_EQUIL_RESTART_ROOT": str(
                (stage / "checkpoints" / "equilibration.checkpoint.*.restart").resolve()
            ),
            "THERMAL_PROD_RESTART_ROOT": str(
                (stage / "checkpoints" / "production.checkpoint.*.restart").resolve()
            ),
            "THERMAL_INITIALIZE_RESTART_ROOT": str(
                (stage / "checkpoints" / "initialize.checkpoint.*.restart").resolve()
            ),
            "THERMAL_EQUIL_FINAL_RESTART": str(
                (stage / "restart.equilibration").resolve()
            ),
            "THERMAL_PRESERVE_EQUIL_FINAL": (
                "yes"
                if checkpoint is not None
                and checkpoint.completed_production_steps > 0
                else "no"
            ),
        }
    )
    if initialization_only:
        resolved_environment.update(
            {
                "THERMAL_INITIALIZE_RESUME": "yes" if checkpoint else "no",
                "THERMAL_INITIAL_NVT_STEPS": str(remaining_prod),
                "THERMAL_INITIALIZE_CSV": str(prod_segment.resolve()),
                "THERMAL_INITIALIZE_FINAL_RESTART": str(final_restart.resolve()),
            }
        )
    else:
        resolved_environment.update(
            {
                "THERMAL_RAMP_START_TEMP_K": str(ramp_start_decimal),
                "THERMAL_RAMP_END_TEMP_K": str(ramp_end_decimal),
            }
        )
    if checkpoint is not None:
        resolved_environment["THERMAL_PREDECESSOR_RESTART"] = str(
            checkpoint.path.resolve()
        )
    elif stage_spec.get("predecessor_restart"):
        resolved_environment["THERMAL_PREDECESSOR_RESTART"] = str(
            _resolve_stage_reference(stage, stage_spec["predecessor_restart"])
        )
    working_directory = _resolve_stage_reference(
        stage, stage_spec.get("working_directory", ".")
    )
    if not working_directory.is_dir():
        raise ThermalSimulationError(
            f"stage working_directory is missing: {working_directory}"
        )
    invocation = StageInvocation(
        command=tuple(str(item) for item in command),
        cwd=working_directory,
        environment=resolved_environment,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        log_path=log_path,
        equilibration_segment_path=equil_segment,
        production_segment_path=prod_segment,
        final_restart_path=final_restart,
        checkpoint_dir=stage / "checkpoints",
        resume_checkpoint=checkpoint,
        remaining_equilibration_steps=remaining_equil,
        remaining_production_steps=remaining_prod,
        segment_index=segment_index,
    )
    _atomic_write_json(
        stage / "stage_status.json",
        {
            "execution_status": "RUNNING",
            "qc_status": "NOT_EVALUATED",
            "analysis_status": "NOT_REQUESTED",
            "segment_index": segment_index,
            "resumed_from": resume_reference,
        },
    )
    runner = process_runner or _default_process_runner
    try:
        returncode = _return_code(runner(invocation))
    except BaseException:
        _atomic_write_json(
            stage / "stage_status.json",
            {
                "execution_status": "INCOMPLETE",
                "qc_status": "NOT_EVALUATED",
                "analysis_status": "NOT_REQUESTED",
                "segment_index": segment_index,
                "resumed_from": resume_reference,
            },
        )
        try:
            promote_stage_boundary_restarts(stage, require_complete=False)
            register_discovered_checkpoints(stage)
        except Exception as checkpoint_error:
            status = _read_json(stage / "stage_status.json")
            status["checkpoint_registration_error"] = (
                f"{type(checkpoint_error).__name__}: {checkpoint_error}"
            )
            _atomic_write_json(stage / "stage_status.json", status)
        raise
    if returncode != 0:
        _atomic_write_json(
            stage / "stage_status.json",
            {
                "execution_status": "INCOMPLETE",
                "qc_status": "NOT_EVALUATED",
                "analysis_status": "NOT_REQUESTED",
                "returncode": returncode,
                "segment_index": segment_index,
                "resumed_from": resume_reference,
            },
        )
        promote_stage_boundary_restarts(stage, require_complete=False)
        register_discovered_checkpoints(stage)
        raise ThermalSimulationError(
            f"stage {stage_spec['stage_id']} exited with code {returncode}"
        )

    promote_stage_boundary_restarts(stage, require_complete=True)
    register_discovered_checkpoints(stage)

    if initialization_only and prod_segment.is_file() and not equil_segment.exists():
        # Initialization has no NPT equilibration window.  Keep the artifact
        # vocabulary uniform by materializing a header-only equilibration
        # segment; it is never eligible for either scientific analysis.
        with prod_segment.open("r", encoding="utf-8", newline="") as source:
            header = source.readline()
        if not header:
            raise StageIntegrityError(
                f"initialization sample segment has no CSV header: {prod_segment}"
            )
        equil_segment.write_text(header, encoding="utf-8")
    required_paths = [
        equil_segment,
        prod_segment,
        final_restart,
        log_path,
        stdout_path,
        stderr_path,
    ]
    equilibration_final_restart = stage / "restart.equilibration"
    if not initialization_only and planned_equil > 0:
        required_paths.append(equilibration_final_restart)
    for required_path in required_paths:
        if not required_path.is_file():
            raise StageIntegrityError(
                f"successful process did not finalize required artifact: {required_path}"
            )
    equil_segments = sorted((stage / "segments").glob("equilibration.segment_*.csv"))
    prod_segments = sorted((stage / "segments").glob("production.segment_*.csv"))
    canonical_equil = stage / "samples" / "equilibration.csv"
    canonical_prod = stage / "samples" / "production.csv"
    (
        equil_minimums,
        equil_cutoffs,
        prod_minimums,
        prod_cutoffs,
        attempt_lineage,
    ) = _attempt_lineage_ranges(
        stage,
        current_attempt_index=segment_index,
        selected_checkpoint=checkpoint,
        planned_equilibration_steps=planned_equil,
        planned_production_steps=planned_prod,
        start_step=int(stage_spec.get("start_step", 0)),
    )
    sample_every_steps = stage_spec.get("sample_every_steps")
    expected_stride = (
        int(sample_every_steps) if sample_every_steps is not None else None
    )
    allowed_equilibration_boundary_duplicates: dict[Path, list[int]] = {}
    for attempt_index in attempt_lineage:
        attempt = _read_json(
            stage / "attempts" / f"attempt_{attempt_index:04d}.json"
        )
        allowed_steps = attempt.get(
            "allowed_equilibration_boundary_duplicate_steps", []
        )
        if not isinstance(allowed_steps, list):
            raise StageIntegrityError(
                "attempt boundary duplicate declaration must be an array"
            )
        if any(
            isinstance(step, bool) or not isinstance(step, int)
            for step in allowed_steps
        ):
            raise StageIntegrityError(
                "attempt boundary duplicate steps must be integers"
            )
        attempt_equil_segment = _resolve_stage_reference(
            stage, attempt["equilibration_segment"]
        )
        allowed_equilibration_boundary_duplicates[attempt_equil_segment] = list(
            allowed_steps
        )
    equil_merge = merge_csv_segments(
        equil_segments,
        canonical_equil,
        minimum_step_exclusive_by_segment=equil_minimums,
        maximum_step_by_segment=equil_cutoffs,
        allowed_boundary_duplicate_steps_by_segment=(
            allowed_equilibration_boundary_duplicates
        ),
        expected_step_stride=expected_stride,
    )
    prod_merge = merge_csv_segments(
        prod_segments,
        canonical_prod,
        minimum_step_exclusive_by_segment=prod_minimums,
        maximum_step_by_segment=prod_cutoffs,
        expected_step_stride=expected_stride,
    )
    qc_production = canonical_prod
    cumulative_production = None
    sampling_sources = stage_spec.get("sampling_production_sources")
    if sampling_sources is not None:
        if (stage_spec.get("sampling_window", {}).get("role") != "density"
                or not isinstance(sampling_sources, list) or not sampling_sources):
            raise StageIntegrityError("cumulative sampling sources are invalid")
        sampling_root = (stage / stage_spec["sampling_run_root_relative"]).resolve()
        verified_sources = [verify_artifact_record(record, relative_to=sampling_root)
                            for record in sampling_sources]
        cumulative_production = stage / "samples" / "production.cumulative.csv"
        merge_csv_segments([*verified_sources, canonical_prod], cumulative_production,
                           expected_step_stride=expected_stride)
        qc_production = cumulative_production
    qc_status = "NOT_EVALUATED"
    qc_artifact: Path | None = None
    if qc_evaluator is not None:
        try:
            qc_result = qc_evaluator(canonical_equil, qc_production)
            if isinstance(qc_result, Mapping):
                qc_payload = dict(qc_result)
                qc_status = str(qc_payload.get("status", "")).upper()
            else:
                qc_status = str(qc_result).upper()
                qc_payload = {"status": qc_status}
            if qc_status not in {"PASS", "FAIL"}:
                raise ThermalSimulationError(
                    "qc_evaluator must return PASS/FAIL or a mapping with that status"
                )
        except Exception as exc:
            # MD artifacts are complete.  A QC implementation failure is a QC
            # failure record, not an excuse to leave execution `.incomplete`.
            qc_status = "FAIL"
            qc_payload = {
                "status": "FAIL",
                "reason_codes": ["QC_EVALUATOR_ERROR"],
                "message": f"{type(exc).__name__}: {exc}",
            }
            required_policies = getattr(
                qc_evaluator, "_thermal_qc_policy_requirements", None
            )
            if isinstance(required_policies, list) and required_policies:
                qc_payload["policy_results"] = [
                    {**dict(item), "status": "FAIL"}
                    for item in required_policies
                    if isinstance(item, Mapping)
                ]
            required_implementation = getattr(
                qc_evaluator, "_thermal_qc_implementation_sha256", None
            )
            if isinstance(required_implementation, str):
                qc_payload["qc_implementation_sha256"] = (
                    required_implementation
                )
        qc_artifact = stage / "qc" / "state_point_qc.json"
        _atomic_write_json(qc_artifact, qc_payload)

    artifact_paths = [
        spec_path,
        log_path,
        stdout_path,
        stderr_path,
        final_restart,
        canonical_equil,
        canonical_prod,
        *equil_segments,
        *prod_segments,
        *sorted((stage / "attempts").glob("attempt_*.json")),
        *sorted((stage / "checkpoints").glob("*")),
    ]
    if not initialization_only and planned_equil > 0:
        artifact_paths.append(equilibration_final_restart)
    if qc_artifact is not None:
        artifact_paths.append(qc_artifact)
    if cumulative_production is not None:
        artifact_paths.append(cumulative_production)
    artifacts = [artifact_record(path, relative_to=stage) for path in artifact_paths]
    manifest = {
        "schema_version": STAGE_MANIFEST_SCHEMA,
        "stage_id": stage_spec["stage_id"],
        "replica_id": stage_spec["replica_id"],
        "stage_spec_sha256": spec_sha256,
        "execution_status": "COMPLETE",
        "qc_status": qc_status,
        "analysis_status": "NOT_REQUESTED",
        "resumed_from": resume_reference,
        "segment_merge": {
            "attempt_lineage": attempt_lineage,
            "equilibration": equil_merge,
            "production": prod_merge,
        },
        "artifacts": artifacts,
    }
    _atomic_write_json(stage / "stage_manifest.json", manifest)
    _atomic_write_json(
        stage / "stage_status.json",
        {
            "execution_status": "COMPLETE",
            "qc_status": qc_status,
            "analysis_status": "NOT_REQUESTED",
            "stage_manifest_sha256": sha256_file(stage / "stage_manifest.json"),
        },
    )
    return StageExecutionResult(
        stage_id=str(stage_spec["stage_id"]),
        stage_dir=stage,
        execution_status="COMPLETE",
        qc_status=qc_status,
        analysis_status="NOT_REQUESTED",
        skipped=False,
        resumed_from=resume_reference,
        manifest_path=stage / "stage_manifest.json",
    )


def finalize_run_directory(
    *,
    incomplete_dir: str | Path,
    final_dir: str | Path,
    run_manifest: Mapping[str, Any],
    stage_results: Sequence[StageExecutionResult],
    initialization_results: Sequence[StageExecutionResult] = (),
) -> Path:
    """Finalize completed execution even when scientific QC has failed."""

    incomplete = Path(incomplete_dir)
    final = Path(final_dir)
    if os.path.lexists(final):
        raise ThermalSimulationError(f"final run directory already exists: {final}")
    if not incomplete.is_dir():
        raise ThermalSimulationError(f"incomplete run directory is missing: {incomplete}")
    if not stage_results or any(item.execution_status != "COMPLETE" for item in stage_results):
        raise ThermalSimulationError("cannot finalize until every planned stage completed")
    if "sampling_continuation" in run_manifest or "sampling_history" in run_manifest:
        from .sampling_continuation import verify_run_sampling_history

        verify_run_sampling_history(incomplete, run_manifest)
    declared_state_points = run_manifest.get("state_points")
    if not isinstance(declared_state_points, list) or not declared_state_points:
        raise ThermalSimulationError(
            "run manifest must declare every planned NPT state point before finalization"
        )
    run_class = run_manifest.get("run_class")
    scientific_eligible = run_manifest.get("scientific_eligible")
    reuse_eligible = run_manifest.get("reuse_eligible")
    if run_class not in RUN_CLASSES or type(scientific_eligible) is not bool or type(
        reuse_eligible
    ) is not bool:
        raise ThermalSimulationError(
            "run manifest lacks explicit run classification and eligibility gates"
        )
    if run_class in NONSCIENTIFIC_RUN_CLASSES and (
        scientific_eligible or reuse_eligible
    ):
        raise ThermalSimulationError(
            f"{run_class} run manifest cannot be scientific or reusable"
        )
    if run_class == "PRODUCTION" and not scientific_eligible:
        raise ThermalSimulationError(
            "PRODUCTION run manifest must be scientifically eligible"
        )
    run_execution_identity = run_manifest.get("execution_identity")
    if not isinstance(run_execution_identity, Mapping):
        raise ThermalSimulationError(
            "run manifest lacks the pinned execution identity"
        )
    try:
        run_density_qc_implementation = density_qc_implementation_sha256(
            run_execution_identity
        )
    except ReuseError as exc:
        raise ThermalSimulationError(
            "run execution identity lacks a valid density-QC implementation"
        ) from exc
    expected_stage_ids: list[str] = []
    expected_spec_by_stage: dict[str, str] = {}
    expected_thermo_by_stage: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    expected_qc_by_stage: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    density_provenance_by_stage: dict[
        str, tuple[Path, StatePointProvenance]
    ] = {}
    for index, state_point in enumerate(declared_state_points):
        if not isinstance(state_point, Mapping):
            raise ThermalSimulationError(f"state_points[{index}] is not an object")
        state_id = state_point.get("state_point_id")
        if not isinstance(state_id, str) or not state_id:
            raise ThermalSimulationError(
                f"state_points[{index}] lacks state_point_id"
            )
        expected_stage_ids.append(state_id)
        if (
            state_point.get("run_class") != run_class
            or state_point.get("scientific_eligible") is not scientific_eligible
            or state_point.get("reuse_eligible") is not reuse_eligible
        ):
            raise StageIntegrityError(
                f"state-point classification mismatch: {state_id}"
            )
        expected_spec = state_point.get("stage_spec_sha256")
        if not isinstance(expected_spec, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_spec
        ):
            raise ThermalSimulationError(
                f"state_points[{index}] lacks a valid stage_spec_sha256"
            )
        expected_spec_by_stage[state_id] = expected_spec
        if (
            state_point.get("qc_implementation_sha256")
            != run_density_qc_implementation
        ):
            raise StageIntegrityError(
                f"state point has an unpinned QC implementation: {state_id}"
            )
        state_provenance_ref = state_point.get("state_point_provenance")
        if state_point.get("use_for_density") is True:
            if not isinstance(state_provenance_ref, Mapping):
                raise ThermalSimulationError(
                    f"density target {state_id} lacks state-point provenance"
                )
            state_provenance_path = verify_artifact_record(
                state_provenance_ref, relative_to=incomplete
            )
            state_provenance_payload = _read_json(state_provenance_path)
            try:
                state_provenance_record = StatePointProvenance.from_mapping(
                    state_provenance_payload
                )
            except (ValueError, TypeError) as exc:
                raise StageIntegrityError(
                    f"invalid state-point provenance: {state_id}"
                ) from exc
            if (
                state_provenance_record.run_id != run_manifest.get("run_id")
                or state_provenance_record.replica_id
                != state_point.get("replica_id")
                or state_provenance_record.state_point_id != state_id
                or state_provenance_record.execution_status
                is not ExecutionStatus.SUCCEEDED
                or state_provenance_payload.get("trajectory_key")
                != state_point.get("trajectory_key")
                or state_provenance_payload.get("density_reuse_key")
                != state_point.get("density_reuse_key")
                or state_provenance_record.run_class != run_class
                or state_provenance_record.scientific_eligible
                is not scientific_eligible
                or state_provenance_record.reuse_eligible is not reuse_eligible
            ):
                raise StageIntegrityError(
                    f"state-point provenance identity mismatch: {state_id}"
                )
            policy_results = state_point.get("qc_policy_results")
            density_policy_results = [
                item
                for item in policy_results
                if isinstance(item, Mapping)
                and item.get("role") == "density_target"
            ] if isinstance(policy_results, list) else []
            if len(density_policy_results) != 1:
                raise StageIntegrityError(
                    f"density target lacks one density QC policy result: {state_id}"
                )
            density_policy_result = density_policy_results[0]
            expected_qc = {
                "PASS": QCStatus.PASSED,
                "FAIL": QCStatus.FAILED,
            }.get(str(density_policy_result.get("status")))
            if (
                expected_qc is None
                or state_provenance_record.qc_status is not expected_qc
                or state_provenance_record.density_qc_policy_id
                != density_policy_result.get("policy_id")
                or state_provenance_record.density_qc_policy_sha256
                != density_policy_result.get("policy_sha256")
                or state_provenance_record.density_qc_implementation_sha256
                != state_point.get("density_qc_implementation_sha256")
                or state_provenance_record.density_qc_implementation_sha256
                != run_density_qc_implementation
            ):
                raise StageIntegrityError(
                    f"state-point provenance density-QC mismatch: {state_id}"
                )
            expected_manifest_path = (
                incomplete / "manifest" / "run_manifest.json"
            ).resolve()
            if (
                state_provenance_path.parent
                / state_provenance_record.run_manifest_relpath
            ).resolve() != expected_manifest_path:
                raise StageIntegrityError(
                    f"state-point provenance manifest lineage mismatch: {state_id}"
                )
            density_provenance_by_stage[state_id] = (
                state_provenance_path,
                state_provenance_record,
            )
        elif (
            state_provenance_ref is not None
            or state_point.get("density_reuse_key") is not None
            or state_point.get("density_qc_implementation_sha256") is not None
        ):
            raise ThermalSimulationError(
                f"non-density state {state_id} cannot advertise density reuse"
            )
        artifacts = state_point.get("artifacts")
        thermo = artifacts.get("thermo_samples") if isinstance(artifacts, Mapping) else None
        state_qc = artifacts.get("state_point_qc") if isinstance(artifacts, Mapping) else None
        if not isinstance(thermo, Mapping):
            raise ThermalSimulationError(
                f"state point {state_id} lacks thermo-sample provenance"
            )
        expected_thermo_by_stage[state_id] = (
            verify_artifact_record(thermo, relative_to=incomplete),
            thermo,
        )
        if not isinstance(state_qc, Mapping):
            raise ThermalSimulationError(
                f"state point {state_id} lacks QC provenance"
            )
        expected_qc_by_stage[state_id] = (
            verify_artifact_record(state_qc, relative_to=incomplete),
            state_qc,
        )
        if state_id in density_provenance_by_stage:
            provenance_path, provenance_record = density_provenance_by_stage[
                state_id
            ]
            provenance_thermo = provenance_record.artifact_named(
                "production_samples"
            )
            provenance_qc = provenance_record.artifact_named(
                "density_qc_result"
            )
            if provenance_thermo is None:
                raise StageIntegrityError(
                    f"density provenance lacks production samples: {state_id}"
                )
            if provenance_qc is None:
                raise StageIntegrityError(
                    f"density provenance lacks density QC output: {state_id}"
                )
            provenance_thermo_path = Path(provenance_thermo.path)
            if not provenance_thermo_path.is_absolute():
                provenance_thermo_path = provenance_path.parent / provenance_thermo_path
            expected_thermo_path = expected_thermo_by_stage[state_id][0]
            if (
                provenance_thermo_path.resolve() != expected_thermo_path.resolve()
                or provenance_thermo.sha256 != thermo.get("sha256")
                or provenance_thermo.size_bytes != thermo.get("bytes")
            ):
                raise StageIntegrityError(
                    f"density provenance/thermo mismatch: {state_id}"
                )
            provenance_qc_path = Path(provenance_qc.path)
            if not provenance_qc_path.is_absolute():
                provenance_qc_path = provenance_path.parent / provenance_qc_path
            expected_qc_path, expected_qc_record = expected_qc_by_stage[state_id]
            if (
                provenance_qc_path.resolve() != expected_qc_path.resolve()
                or provenance_qc.sha256 != expected_qc_record.get("sha256")
                or provenance_qc.size_bytes != expected_qc_record.get("bytes")
            ):
                raise StageIntegrityError(
                    f"density provenance/QC mismatch: {state_id}"
                )
    actual_stage_ids = [item.stage_id for item in stage_results]
    if len(set(expected_stage_ids)) != len(expected_stage_ids):
        raise ThermalSimulationError("run manifest contains duplicate state_point_id values")
    if len(set(actual_stage_ids)) != len(actual_stage_ids):
        raise ThermalSimulationError("stage results contain duplicate stage IDs")
    if set(actual_stage_ids) != set(expected_stage_ids):
        raise ThermalSimulationError(
            "completed-stage coverage does not match the resolved state-point plan: "
            f"expected={sorted(expected_stage_ids)}, actual={sorted(actual_stage_ids)}"
        )
    authenticated_qc_values: list[str] = []
    state_by_id = {
        str(item["state_point_id"]): item for item in declared_state_points
    }
    for result in stage_results:
        if result.manifest_path is None:
            raise ThermalSimulationError(
                f"completed stage lacks a manifest: {result.stage_id}"
            )
        try:
            result.stage_dir.resolve().relative_to(incomplete.resolve())
        except ValueError as exc:
            raise StageIntegrityError(
                f"stage result escapes the incomplete run: {result.stage_id}"
            ) from exc
        canonical_manifest_path = result.stage_dir / "stage_manifest.json"
        if result.manifest_path.resolve() != canonical_manifest_path.resolve():
            raise StageIntegrityError(
                f"stage result references a noncanonical manifest: {result.stage_id}"
            )
        manifest = _read_json(canonical_manifest_path)
        if manifest.get("stage_id") != result.stage_id:
            raise StageIntegrityError(
                f"stage result/manifest identity mismatch: {result.stage_id}"
            )
        manifest_execution = manifest.get("execution_status")
        manifest_qc = manifest.get("qc_status")
        manifest_analysis = manifest.get("analysis_status")
        if (
            manifest_execution != result.execution_status
            or manifest_qc != result.qc_status
            or manifest_analysis != result.analysis_status
        ):
            raise StageIntegrityError(
                f"stage result/status mismatch: {result.stage_id}"
            )
        state_record = state_by_id[result.stage_id]
        if (
            state_record.get("status") != manifest_execution
            or state_record.get("qc_status") != manifest_qc
        ):
            raise StageIntegrityError(
                f"run state-point status mismatch: {result.stage_id}"
            )
        if not _stage_is_reusable(
            result.stage_dir, expected_spec_by_stage[result.stage_id]
        ):
            raise StageIntegrityError(
                f"stage failed final integrity validation: {result.stage_id}"
            )
        expected_thermo_path, expected_thermo_record = expected_thermo_by_stage[
            result.stage_id
        ]
        selected_spec = _read_json(result.stage_dir / "stage_spec.json")
        selected_production_name = ("production.cumulative.csv"
            if selected_spec.get("sampling_production_sources") else "production.csv")
        canonical_production_path = (
            result.stage_dir / "samples" / selected_production_name
        ).resolve()
        if expected_thermo_path.resolve() != canonical_production_path:
            raise StageIntegrityError(
                f"run thermo artifact is not canonical production data: {result.stage_id}"
            )
        manifest_artifacts = manifest.get("artifacts")
        if not isinstance(manifest_artifacts, list):
            raise StageIntegrityError(
                f"stage manifest lacks artifact inventory: {result.stage_id}"
            )
        matching_stage_thermo: list[Mapping[str, Any]] = []
        matching_qc_artifacts: list[Mapping[str, Any]] = []
        canonical_qc_path = (
            result.stage_dir / "qc" / "state_point_qc.json"
        ).resolve()
        expected_qc_path, expected_qc_record = expected_qc_by_stage[
            result.stage_id
        ]
        if expected_qc_path.resolve() != canonical_qc_path:
            raise StageIntegrityError(
                f"run QC artifact is not canonical state-point QC: {result.stage_id}"
            )
        for artifact in manifest_artifacts:
            if not isinstance(artifact, Mapping):
                continue
            verified = verify_artifact_record(
                artifact, relative_to=result.stage_dir
            )
            if verified.resolve() == expected_thermo_path.resolve():
                matching_stage_thermo.append(artifact)
            if verified.resolve() == canonical_qc_path:
                matching_qc_artifacts.append(artifact)
        if len(matching_stage_thermo) != 1:
            raise StageIntegrityError(
                f"run thermo artifact is not bound to one stage artifact: {result.stage_id}"
            )
        stage_thermo_record = matching_stage_thermo[0]
        if (
            stage_thermo_record.get("sha256")
            != expected_thermo_record.get("sha256")
            or stage_thermo_record.get("bytes")
            != expected_thermo_record.get("bytes")
        ):
            raise StageIntegrityError(
                f"run/stage thermo provenance mismatch: {result.stage_id}"
            )
        if manifest_qc in {"PASS", "FAIL"}:
            if len(matching_qc_artifacts) != 1:
                raise StageIntegrityError(
                    f"stage QC status lacks one canonical QC artifact: {result.stage_id}"
                )
            stage_qc_record = matching_qc_artifacts[0]
            if (
                stage_qc_record.get("sha256")
                != expected_qc_record.get("sha256")
                or stage_qc_record.get("bytes")
                != expected_qc_record.get("bytes")
            ):
                raise StageIntegrityError(
                    f"run/stage QC provenance mismatch: {result.stage_id}"
                )
            qc_payload = _read_json(canonical_qc_path)
            if str(qc_payload.get("status", "")).upper() != manifest_qc:
                raise StageIntegrityError(
                    f"stage QC artifact/status mismatch: {result.stage_id}"
                )
            if (
                qc_payload.get("qc_implementation_sha256")
                != run_density_qc_implementation
                or qc_payload.get("qc_implementation_sha256")
                != state_record.get("qc_implementation_sha256")
            ):
                raise StageIntegrityError(
                    f"stage QC implementation mismatch: {result.stage_id}"
                )
            authenticated_policy_results = _policy_result_summary(qc_payload)
            if canonical_sha256(authenticated_policy_results) != canonical_sha256(
                state_record.get("qc_policy_results")
            ):
                raise StageIntegrityError(
                    f"run/stage QC policy results mismatch: {result.stage_id}"
                )
        elif manifest_qc == "NOT_EVALUATED":
            if matching_qc_artifacts:
                raise StageIntegrityError(
                    f"unevaluated stage unexpectedly has QC output: {result.stage_id}"
                )
        else:
            raise StageIntegrityError(
                f"unsupported stage QC status: {result.stage_id}={manifest_qc}"
            )
        authenticated_qc_values.append(str(manifest_qc))
    declared_initializers = run_manifest.get("initialization_stages", [])
    if declared_initializers or initialization_results:
        if not isinstance(declared_initializers, list):
            raise ThermalSimulationError("initialization_stages must be an array")
        expected_initializers: dict[str, Mapping[str, Any]] = {}
        for index, item in enumerate(declared_initializers):
            if not isinstance(item, Mapping):
                raise ThermalSimulationError(
                    f"initialization_stages[{index}] is not an object"
                )
            stage_id = item.get("stage_id")
            spec_hash = item.get("stage_spec_sha256")
            if not isinstance(stage_id, str) or not isinstance(spec_hash, str):
                raise ThermalSimulationError(
                    f"initialization_stages[{index}] lacks identity"
                )
            expected_initializers[stage_id] = item
        actual_initializers = {item.stage_id: item for item in initialization_results}
        if set(expected_initializers) != set(actual_initializers):
            raise ThermalSimulationError(
                "initializer coverage does not match the resolved replica plan"
            )
        for stage_id, result in actual_initializers.items():
            declared = expected_initializers[stage_id]
            expected_spec_sha256 = str(declared["stage_spec_sha256"])
            if result.execution_status != "COMPLETE" or not _stage_is_reusable(
                result.stage_dir, expected_spec_sha256
            ):
                raise StageIntegrityError(
                    f"initializer failed final integrity validation: {stage_id}"
                )
            initializer_manifest = _read_json(result.stage_dir / "stage_manifest.json")
            manifest_qc = str(
                initializer_manifest.get("qc_status", "NOT_EVALUATED")
            ).upper()
            if result.qc_status != manifest_qc:
                raise StageIntegrityError(
                    f"initializer result/manifest QC mismatch: {stage_id}"
                )
            declared_qc = str(
                declared.get("qc_status", "NOT_EVALUATED")
            ).upper()
            if declared_qc != manifest_qc:
                raise StageIntegrityError(
                    f"run/initializer QC mismatch: {stage_id}"
                )
            if manifest_qc in {"PASS", "FAIL"}:
                authenticated_qc_values.append(manifest_qc)
    qc_values = authenticated_qc_values
    qc_status = (
        "FAIL"
        if "FAIL" in qc_values
        else (
            "PASS"
            if qc_values and all(value == "PASS" for value in qc_values)
            else "NOT_EVALUATED"
        )
    )
    status = "COMPLETE_WITH_QC_FAILURE" if qc_status == "FAIL" else "COMPLETE"
    payload = dict(run_manifest)
    payload.update(
        {
            "schema_version": RUN_MANIFEST_SCHEMA,
            "status": status,
            "finalized": True,
            "execution_status": "COMPLETE",
            "qc_status": qc_status,
            "analysis_status": payload.get("analysis_status", "NOT_REQUESTED"),
            "stage_manifests": [
                artifact_record(item.manifest_path, relative_to=incomplete)
                for item in stage_results
                if item.manifest_path is not None
            ],
        }
    )
    manifest_path = incomplete / "manifest" / "run_manifest.json"
    _atomic_write_json(manifest_path, payload)
    digest_path = incomplete / "manifest" / "run_manifest.sha256"
    digest_path.write_text(sha256_file(manifest_path) + "\n", encoding="ascii")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(incomplete, final)
    return final / "manifest" / "run_manifest.json"


def validate_density_continuation(
    *,
    current_total_steps: int,
    requested_additional_steps: int,
    maximum_total_steps: int,
) -> dict[str, int | str]:
    """Validate an explicit bounded density-only continuation request.

    Nothing calls this automatically.  Tg stages are deliberately ineligible
    for continuation because changing one dwell would change their protocol.
    """

    values = (current_total_steps, requested_additional_steps, maximum_total_steps)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ThermalSimulationError("density continuation limits must be integers")
    if current_total_steps < 0 or requested_additional_steps <= 0 or maximum_total_steps <= 0:
        raise ThermalSimulationError("density continuation limits must be positive")
    new_total = current_total_steps + requested_additional_steps
    if new_total > maximum_total_steps:
        raise ThermalSimulationError(
            f"requested density continuation reaches {new_total} steps, above "
            f"configured maximum {maximum_total_steps}"
        )
    return {
        "schema_version": "thermal-properties-density-continuation-plan/v1",
        "mode": "density_explicit_continuation",
        "automatic": False,
        "execution_status": "PLANNED",
        "qc_status": "NOT_EVALUATED",
        "analysis_status": "NOT_REQUESTED",
        "current_total_steps": current_total_steps,
        "requested_additional_steps": requested_additional_steps,
        "maximum_total_steps": maximum_total_steps,
        "new_total_steps": new_total,
    }


def create_incomplete_run_directory(output_root: str | Path, run_id: str | None = None) -> tuple[str, Path]:
    """Allocate ``<run_id>.incomplete`` without a latest alias or overwrite."""

    identifier = _validate_run_id(run_id or f"run_{uuid.uuid4()}")
    destination = Path(output_root).resolve() / f"{identifier}.incomplete"
    destination.mkdir(parents=True, exist_ok=False)
    return identifier, destination


def _validate_run_id(value: object) -> str:
    try:
        return validate_run_id(value)
    except ClaimValidationError as exc:
        raise ThermalSimulationError(str(exc)) from exc


def copy_runtime_input(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Copy one immutable runtime input without editing its source."""

    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise ThermalSimulationError(f"runtime input is missing: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb") as reader, destination_path.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
    return {
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "copied_path": str(destination_path.resolve()),
        "copied_sha256": sha256_file(destination_path),
    }


def _copy_or_verify_runtime_input(
    source: str | Path, destination: str | Path
) -> dict[str, Any]:
    source_path = Path(source)
    destination_path = Path(destination)
    if destination_path.exists():
        if not source_path.is_file() or not destination_path.is_file():
            raise StageIntegrityError(
                f"runtime input cannot be verified on resume: {destination_path}"
            )
        source_hash = sha256_file(source_path)
        copied_hash = sha256_file(destination_path)
        if source_hash != copied_hash:
            raise StageIntegrityError(
                f"snapshotted runtime input changed: {destination_path}"
            )
        return {
            "source_path": str(source_path.resolve()),
            "source_sha256": source_hash,
            "copied_path": str(destination_path.resolve()),
            "copied_sha256": copied_hash,
        }
    return copy_runtime_input(source_path, destination_path)


def _campaign_stage_environment(
    resolved: Mapping[str, Any],
    *,
    input_data: Path,
    stage_dir: Path,
) -> dict[str, str]:
    engine = _require_mapping(resolved.get("engine"), "resolved.engine")
    system = _require_mapping(resolved.get("system"), "resolved.system")
    initialize = _require_mapping(resolved.get("initialize"), "resolved.initialize")
    npt = _require_mapping(resolved.get("npt"), "resolved.npt")
    timestep = engine["dt_ps"]
    elements = _require_sequence(system.get("mace_elements"), "system.mace_elements")
    return {
        "THERMAL_INPUT_DATA": str(input_data.resolve()),
        "THERMAL_MACE_MODEL": str(Path(str(system["mace_model"])).resolve()),
        "THERMAL_MACE_ELEMENTS": " ".join(str(value) for value in elements),
        "THERMAL_DT_PS": str(timestep),
        "THERMAL_TDAMP_PS": str(engine["tdamp_ps"]),
        "THERMAL_PDAMP_PS": str(engine["pdamp_ps"]),
        "THERMAL_INITIAL_TEMP_K": str(initialize["temperature_k"]),
        "THERMAL_MINIMIZE": "yes" if initialize["minimize"] else "no",
        "THERMAL_MIN_ETOL": str(initialize.get("min_etol", 1e-8)),
        "THERMAL_MIN_FTOL": str(initialize.get("min_ftol_eV_A", 1e-10)),
        "THERMAL_MIN_MAXITER": str(initialize.get("min_maxiter", 2000)),
        "THERMAL_MIN_MAXEVAL": str(initialize.get("min_maxeval", 20000)),
        "THERMAL_THERMO_EVERY_STEPS": str(
            ps_to_steps(npt["thermo_interval_ps"], timestep)
        ),
        "THERMAL_SAMPLE_EVERY_STEPS": str(
            ps_to_steps(npt["sample_interval_ps"], timestep)
        ),
        "THERMAL_RESTART_EVERY_STEPS": str(
            ps_to_steps(npt["restart_interval_ps"], timestep)
        ),
        "THERMAL_INITIALIZE_RESTART_ROOT": str(
            (stage_dir / "checkpoints" / "initialize.checkpoint.*.restart").resolve()
        ),
        "THERMAL_EQUIL_RESTART_ROOT": str(
            (stage_dir / "checkpoints" / "equilibration.checkpoint.*.restart").resolve()
        ),
        "THERMAL_PROD_RESTART_ROOT": str(
            (stage_dir / "checkpoints" / "production.checkpoint.*.restart").resolve()
        ),
        "THERMAL_EQUIL_FINAL_RESTART": str(
            (stage_dir / "restart.equilibration").resolve()
        ),
    }


def _density_qc_policy_identity() -> dict[str, str]:
    document = _load_json_object(
        Path(__file__).resolve().parent / "config" / "density_qc_v1.json"
    )
    policy_id = str(document.get("policy_id", ""))
    if not policy_id:
        raise ThermalSimulationError("density QC policy lacks policy_id")
    return {
        "policy_id": policy_id,
        "policy_sha256": canonical_sha256(document),
    }


def _statepoint_qc_policy_requirements(
    stage_spec: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Resolve every role-specific QC policy required by one state point."""

    package_root = Path(__file__).resolve().parent
    state = _require_mapping(stage_spec.get("state"), "stage_spec.state")
    requirements: list[dict[str, Any]] = []
    if stage_spec.get("stage_role") == "mace_transition_npt":
        document = _load_json_object(
            package_root / "config" / "mace_transition_qc_v1.json"
        )
        requirements.append(
            {
                "role": "mace_transition",
                "policy_id": str(document.get("policy_id")),
                "policy_sha256": canonical_sha256(document),
                "policy": dict(document),
            }
        )
    if state.get("use_for_tg_fit") is True:
        document = _load_json_object(package_root / "config" / "tg_fit_v1.json")
        policy = _require_mapping(
            document.get("state_point_qc"), "tg_fit_v1.state_point_qc"
        )
        requirements.append(
            {
                "role": "tg_fit",
                "policy_id": f"{document.get('policy_id')}/state-point-qc",
                "policy_sha256": canonical_sha256(policy),
                "policy": dict(policy),
            }
        )
    if state.get("use_for_density") is True:
        document = _load_json_object(
            package_root / "config" / "density_qc_v1.json"
        )
        density_identity = _density_qc_policy_identity()
        requirements.append(
            {
                "role": "density_target",
                **density_identity,
                "policy": dict(document),
            }
        )
    if not requirements and stage_spec.get("branch") == "cooling":
        # A schedule point may be required only to preserve the declared
        # cooling history.  It is still checked for trajectory/convergence
        # stability, but this role never makes it eligible for the Tg fit.
        document = _load_json_object(package_root / "config" / "tg_fit_v1.json")
        policy = _require_mapping(
            document.get("state_point_qc"), "tg_fit_v1.state_point_qc"
        )
        requirements.append(
            {
                "role": "thermal_history",
                "policy_id": f"{document.get('policy_id')}/state-point-qc",
                "policy_sha256": canonical_sha256(policy),
                "policy": dict(policy),
            }
        )
    if not requirements:
        raise ThermalSimulationError(
            "every production state point must have a declared analysis or "
            "thermal-history role"
        )
    return tuple(requirements)


def _policy_result_summary(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    """Validate and reduce the authenticated role-specific QC results."""

    raw_results = payload.get("policy_results")
    if not isinstance(raw_results, list) or not raw_results:
        raise StageIntegrityError("state-point QC lacks role-specific policy results")
    summaries: list[dict[str, str]] = []
    roles: set[str] = set()
    for index, raw in enumerate(raw_results):
        if not isinstance(raw, Mapping):
            raise StageIntegrityError(
                f"state-point QC policy_results[{index}] is not an object"
            )
        role = str(raw.get("role", ""))
        policy_id = str(raw.get("policy_id", ""))
        policy_sha256 = str(raw.get("policy_sha256", ""))
        status = str(raw.get("status", "")).upper()
        if (
            not role
            or role in roles
            or not policy_id
            or not re.fullmatch(r"[0-9a-f]{64}", policy_sha256)
            or status not in {"PASS", "FAIL"}
        ):
            raise StageIntegrityError(
                f"invalid role-specific QC identity at policy_results[{index}]"
            )
        roles.add(role)
        summaries.append(
            {
                "role": role,
                "policy_id": policy_id,
                "policy_sha256": policy_sha256,
                "status": status,
            }
        )
    overall = str(payload.get("status", "")).upper()
    expected_overall = (
        "PASS" if all(item["status"] == "PASS" for item in summaries) else "FAIL"
    )
    if overall != expected_overall:
        raise StageIntegrityError(
            "state-point QC overall status disagrees with its policy results"
        )
    return summaries


def _bind_statepoint_qc_evaluator(
    evaluator: Callable[[Path, Path], str | Mapping[str, Any]],
    requirements: Sequence[Mapping[str, Any]],
    *,
    qc_implementation_sha256: str,
) -> Callable[[Path, Path], Mapping[str, Any]]:
    """Require an injected/default evaluator to attest every declared role."""

    expected = [
        {
            "role": str(item["role"]),
            "policy_id": str(item["policy_id"]),
            "policy_sha256": str(item["policy_sha256"]),
        }
        for item in requirements
    ]
    if not re.fullmatch(r"[0-9a-f]{64}", qc_implementation_sha256):
        raise ThermalSimulationError("invalid density-QC implementation hash")

    def evaluate(equilibration: Path, production: Path) -> Mapping[str, Any]:
        raw = evaluator(equilibration, production)
        if isinstance(raw, Mapping):
            payload = dict(raw)
            status = str(payload.get("status", "")).upper()
        else:
            status = str(raw).upper()
            payload = {"status": status}
        if status not in {"PASS", "FAIL"}:
            raise ThermalSimulationError(
                "state-point QC evaluator must return PASS or FAIL"
            )
        if "policy_results" not in payload:
            payload["policy_results"] = [
                {**item, "status": status} for item in expected
            ]
        summaries = _policy_result_summary(payload)
        actual_identities = [
            {
                "role": item["role"],
                "policy_id": item["policy_id"],
                "policy_sha256": item["policy_sha256"],
            }
            for item in summaries
        ]
        if actual_identities != expected:
            raise ThermalSimulationError(
                "state-point QC evaluator used unexpected role/policy identities"
            )
        payload.setdefault(
            "schema_version", "thermal-properties-state-point-qc/v1"
        )
        declared_implementation = payload.setdefault(
            "qc_implementation_sha256", qc_implementation_sha256
        )
        if declared_implementation != qc_implementation_sha256:
            raise ThermalSimulationError(
                "state-point QC evaluator used unexpected implementation identity"
            )
        return payload

    setattr(evaluate, "_thermal_qc_policy_requirements", expected)
    setattr(
        evaluate,
        "_thermal_qc_implementation_sha256",
        qc_implementation_sha256,
    )
    return evaluate


def _default_statepoint_qc_evaluator(
    resolved: Mapping[str, Any], stage_spec: Mapping[str, Any]
) -> Callable[[Path, Path], Mapping[str, Any]]:
    """Build deterministic, role-aware QC used before run finalization."""

    requirements = _statepoint_qc_policy_requirements(stage_spec)
    execution_identity = _require_mapping(
        resolved.get("execution_identity"), "resolved.execution_identity"
    )
    qc_implementation_sha256 = density_qc_implementation_sha256(
        execution_identity
    )
    state = _require_mapping(stage_spec.get("state"), "stage_spec.state")
    target_temperature = float(state["temperature_start_k"])
    target_pressure = float(state["pressure_start_bar"])
    replica_id = str(stage_spec.get("replica_id", ""))
    snapshot_identity = _require_mapping(
        _require_mapping(
            execution_identity.get("input_snapshots"),
            "execution_identity.input_snapshots",
        ).get(replica_id),
        f"execution_identity.input_snapshots.{replica_id}",
    )
    expected_atom_count = int(snapshot_identity["atom_count"])

    # Import after execution-identity verification.  The function object and
    # parsed policy documents then remain fixed throughout the MD stage.
    import numpy as np
    import pandas as pd

    from .analysis.convergence import assess_density_convergence

    def evaluate(_equilibration: Path, production: Path) -> Mapping[str, Any]:
        frame = pd.read_csv(production)
        required_diagnostics = (
            "pe_eV",
            "ke_eV",
            "etotal_eV",
            "fmax_eV_A",
        )
        missing = [column for column in required_diagnostics if column not in frame]
        diagnostics: dict[str, Any] = {
            "required_columns": list(required_diagnostics),
            "missing_columns": missing,
        }
        finite_diagnostics = False
        if not missing and not frame.empty:
            diagnostic_values = frame[list(required_diagnostics)].apply(
                pd.to_numeric, errors="coerce"
            )
            finite_diagnostics = bool(
                np.isfinite(diagnostic_values.to_numpy(dtype=float)).all()
            )
            diagnostics.update(
                {
                    "all_finite": finite_diagnostics,
                    "maximum_force_eV_A": float(
                        diagnostic_values["fmax_eV_A"].max()
                    ),
                    "minimum_total_energy_eV": float(
                        diagnostic_values["etotal_eV"].min()
                    ),
                    "maximum_total_energy_eV": float(
                        diagnostic_values["etotal_eV"].max()
                    ),
                }
            )
        else:
            diagnostics["all_finite"] = False
        if "density_g_cm3" in frame:
            with np.errstate(divide="ignore", invalid="ignore"):
                frame["specific_volume_cm3_g"] = 1.0 / pd.to_numeric(
                    frame["density_g_cm3"], errors="coerce"
                ).to_numpy(dtype=float)

        policy_results: list[dict[str, Any]] = []
        for requirement in requirements:
            convergence = assess_density_convergence(
                frame,
                target_temperature_K=target_temperature,
                target_pressure_bar=target_pressure,
                policy=_require_mapping(
                    requirement.get("policy"), "state-point QC policy"
                ),
                expected_atom_count=expected_atom_count,
            )
            status = (
                "PASS"
                if convergence.get("status") == "PASS" and finite_diagnostics
                else "FAIL"
            )
            policy_results.append(
                {
                    "role": requirement["role"],
                    "policy_id": requirement["policy_id"],
                    "policy_sha256": requirement["policy_sha256"],
                    "status": status,
                    "convergence": convergence,
                }
            )
        status = (
            "PASS"
            if all(item["status"] == "PASS" for item in policy_results)
            else "FAIL"
        )
        return {
            "schema_version": "thermal-properties-state-point-qc/v1",
            "status": status,
            "target_temperature_K": target_temperature,
            "target_pressure_bar": target_pressure,
            "diagnostic_observables": diagnostics,
            "policy_results": policy_results,
        }

    return _bind_statepoint_qc_evaluator(
        evaluate,
        requirements,
        qc_implementation_sha256=qc_implementation_sha256,
    )


def _resolve_command_file(token: str, *, primary: bool) -> Path | None:
    """Resolve a command token when it names an executable or file.

    The complete argv is always part of provenance.  In addition, every token
    that resolves to a regular file is content-hashed.  The primary command is
    required for real execution/reuse planning, but synthetic process runners
    may deliberately use an unresolvable placeholder.
    """

    candidate: Path | None = None
    if Path(token).is_absolute() or os.sep in token:
        candidate = Path(token).expanduser().resolve()
    else:
        located = shutil.which(token)
        if located:
            candidate = Path(located).resolve()
    if candidate is not None and candidate.is_file():
        return candidate
    if primary:
        raise ThermalSimulationError(
            f"LAMMPS command executable is not a regular file: {token}"
        )
    return None


def build_engine_identity(
    resolved: Mapping[str, Any], *, require_primary_executable: bool = True
) -> dict[str, Any]:
    """Pin the configured command argv and all command tokens that are files."""

    engine = _require_mapping(resolved.get("engine"), "resolved.engine")
    raw_command = _require_sequence(
        engine.get("lammps_command"), "engine.lammps_command"
    )
    if not raw_command or not all(
        isinstance(item, str) and item for item in raw_command
    ):
        raise ThermalSimulationError(
            "engine.lammps_command must contain nonempty command arguments"
        )
    command = [str(item) for item in raw_command]
    command_files: list[dict[str, Any]] = []
    for index, token in enumerate(command):
        resolved_file = _resolve_command_file(
            token,
            primary=index == 0 and require_primary_executable,
        )
        if resolved_file is None:
            continue
        command_files.append(
            {
                "argv_index": index,
                "token": token,
                "resolved_path": str(resolved_file),
                "sha256": sha256_file(resolved_file),
                "bytes": resolved_file.stat().st_size,
            }
        )
    raw_dependencies = _require_sequence(
        engine.get("runtime_dependencies"), "engine.runtime_dependencies"
    )
    dependency_files: list[dict[str, Any]] = []
    for raw_dependency in raw_dependencies:
        dependency = Path(str(raw_dependency)).expanduser().resolve()
        if not dependency.is_file():
            if require_primary_executable:
                raise ThermalSimulationError(
                    f"engine runtime dependency is not a regular file: {dependency}"
                )
            continue
        dependency_files.append(
            {
                "declared_path": str(raw_dependency),
                "resolved_path": str(dependency),
                "sha256": sha256_file(dependency),
                "bytes": dependency.stat().st_size,
            }
        )
    launch_argv = list(command)
    primary_records = [
        record for record in command_files if record["argv_index"] == 0
    ]
    if primary_records:
        launch_argv[0] = str(primary_records[0]["resolved_path"])
    return {
        "name": engine.get("name"),
        "device": engine.get("device"),
        "command_argv": command,
        "launch_argv": launch_argv,
        "command_files": command_files,
        "declared_runtime_dependencies": [
            str(item) for item in raw_dependencies
        ],
        "runtime_dependency_files": dependency_files,
        "primary_executable_verified": bool(
            command_files and command_files[0]["argv_index"] == 0
        ),
    }


def build_execution_identity(
    resolved: Mapping[str, Any], *, require_primary_executable: bool = True
) -> dict[str, Any]:
    """Hash every external input needed before a campaign creates output."""

    system = _require_mapping(resolved.get("system"), "resolved.system")
    model_path = Path(str(system.get("mace_model", ""))).expanduser().resolve()
    if not model_path.is_file():
        raise ThermalSimulationError(f"MACE model is missing: {model_path}")
    model_sha256 = sha256_file(model_path)
    raw_replicas = _require_sequence(resolved.get("replicas"), "resolved.replicas")
    input_snapshots: dict[str, dict[str, Any]] = {}
    for index, raw_replica in enumerate(raw_replicas):
        replica = _require_mapping(raw_replica, f"resolved.replicas[{index}]")
        replica_id = str(replica.get("replica_id", ""))
        if not replica_id or replica_id in input_snapshots:
            raise ThermalSimulationError("resolved replicas have invalid identities")
        try:
            contract = load_snapshot_contract(
                snapshot_path=str(replica.get("input_data", "")),
                metadata_path=str(replica.get("snapshot_metadata", "")),
                expected_class=str(replica.get("snapshot_class", "")),
                mace_model_sha256=model_sha256,
            )
        except SnapshotContractError as exc:
            raise ThermalSimulationError(
                f"snapshot contract failed for {replica_id}: {exc}"
            ) from exc
        if contract.atom_type_count != len(system.get("mace_elements", [])):
            raise ThermalSimulationError(
                f"snapshot atom-type count disagrees with mace_elements for {replica_id}"
            )
        if set(contract.element_counts) != set(system.get("element_list", [])):
            raise ThermalSimulationError(
                f"snapshot element counts disagree with element_list for {replica_id}"
            )
        input_snapshots[replica_id] = contract.to_dict()
    if not input_snapshots:
        raise ThermalSimulationError("execution requires at least one input snapshot")
    contracts = list(input_snapshots.values())
    composition_keys = {
        canonical_sha256(
            {
                "atom_count": item["atom_count"],
                "atom_type_count": item["atom_type_count"],
                "element_counts": item["element_counts"],
            }
        )
        for item in contracts
    }
    if len(composition_keys) != 1:
        raise ThermalSimulationError(
            "replica snapshots must have identical atom and element composition"
        )
    if resolved.get("run_class") == "PRODUCTION":
        for field in ("snapshot_sha256", "packing_id"):
            values = [str(item[field]) for item in contracts]
            if len(set(values)) != len(values):
                raise ThermalSimulationError(
                    f"PRODUCTION packing replicas do not have distinct {field} values"
                )
    first_snapshot = contracts[0]
    package_root = Path(__file__).resolve().parent
    runtime_hashes: dict[str, str] = {}
    for name in RUNTIME_INPUT_FILES:
        source = package_root / "lammps" / name
        if not source.is_file():
            raise ThermalSimulationError(f"thermal runtime input is missing: {source}")
        runtime_hashes[name] = sha256_file(source)
    orchestration_hashes: dict[str, str] = {}
    for name in ORCHESTRATION_SOURCE_FILES:
        source = package_root / name
        if not source.is_file():
            raise ThermalSimulationError(
                f"thermal orchestration source is missing: {source}"
            )
        orchestration_hashes[name] = sha256_file(source)
    quality_control_source_hashes: dict[str, str] = {}
    for name in QUALITY_CONTROL_SOURCE_FILES:
        source = package_root / name
        if not source.is_file():
            raise ThermalSimulationError(
                f"thermal QC source is missing: {source}"
            )
        quality_control_source_hashes[name] = sha256_file(source)
    quality_control_policy_hashes: dict[str, str] = {}
    for name in QUALITY_CONTROL_POLICY_FILES:
        source = package_root / name
        if not source.is_file():
            raise ThermalSimulationError(
                f"thermal QC policy is missing: {source}"
            )
        quality_control_policy_hashes[name] = sha256_file(source)
    identity = {
        "model": {
            "path": str(model_path),
            "sha256": model_sha256,
            "bytes": model_path.stat().st_size,
            "head": system.get("mace_head"),
            "dtype": system.get("mace_dtype"),
        },
        "input_snapshot": {
            "path": first_snapshot["snapshot_path"],
            "sha256": first_snapshot["snapshot_sha256"],
            "bytes": first_snapshot["snapshot_bytes"],
        },
        "input_snapshot_sha256": first_snapshot["snapshot_sha256"],
        "input_snapshots": input_snapshots,
        "runtime_input_sha256": runtime_hashes,
        "orchestration_source_sha256": orchestration_hashes,
        "quality_control_source_sha256": quality_control_source_hashes,
        "quality_control_policy_sha256": quality_control_policy_hashes,
        "engine": build_engine_identity(
            resolved,
            require_primary_executable=require_primary_executable,
        ),
    }
    parent = resolved.get("density_parent_restart")
    if parent is not None:
        identity["density_parent_restart"] = verify_density_parent(
            parent, resolved, identity
        )
    return identity


def _verify_execution_identity(resolved: Mapping[str, Any], input_root: Path) -> None:
    identity = _require_mapping(
        resolved.get("execution_identity"), "resolved.execution_identity"
    )
    parent = identity.get("density_parent_restart")
    if parent is not None:
        for key, record in parent["source_files"].items():
            candidate = _density_parent_copy_path(input_root, key)
            if (
                candidate.is_symlink()
                or not candidate.is_file()
                or candidate.stat().st_size != record["bytes"]
                or sha256_file(candidate) != record["sha256"]
            ):
                raise StageIntegrityError(f"copied parent evidence changed: {key}")
    model = _require_mapping(identity.get("model"), "execution_identity.model")
    model_path = Path(str(model.get("path", "")))
    if (
        not model_path.is_file()
        or model_path.stat().st_size != int(model.get("bytes", -1))
        or sha256_file(model_path) != model.get("sha256")
    ):
        raise StageIntegrityError(
            "pinned MACE model changed or disappeared before stage launch"
        )
    runtime_hashes = _require_mapping(
        identity.get("runtime_input_sha256"),
        "execution_identity.runtime_input_sha256",
    )
    for name, expected_hash in runtime_hashes.items():
        candidate = input_root / str(name)
        if not candidate.is_file() or sha256_file(candidate) != expected_hash:
            raise StageIntegrityError(
                f"snapshotted runtime input changed before stage launch: {candidate}"
            )
    orchestration_hashes = _require_mapping(
        identity.get("orchestration_source_sha256"),
        "execution_identity.orchestration_source_sha256",
    )
    package_root = Path(__file__).resolve().parent
    for name, expected_hash in orchestration_hashes.items():
        candidate = package_root / str(name)
        if not candidate.is_file() or sha256_file(candidate) != expected_hash:
            raise StageIntegrityError(
                f"thermal orchestration source changed during execution: {candidate}"
            )
    quality_control_source_hashes = _require_mapping(
        identity.get("quality_control_source_sha256"),
        "execution_identity.quality_control_source_sha256",
    )
    for name, expected_hash in quality_control_source_hashes.items():
        candidate = package_root / str(name)
        if not candidate.is_file() or sha256_file(candidate) != expected_hash:
            raise StageIntegrityError(
                f"thermal QC source changed during execution: {candidate}"
            )
    quality_control_policy_hashes = _require_mapping(
        identity.get("quality_control_policy_sha256"),
        "execution_identity.quality_control_policy_sha256",
    )
    for name, expected_hash in quality_control_policy_hashes.items():
        candidate = package_root / str(name)
        if not candidate.is_file() or sha256_file(candidate) != expected_hash:
            raise StageIntegrityError(
                f"thermal QC policy changed during execution: {candidate}"
            )
    input_snapshots = _require_mapping(
        identity.get("input_snapshots"), "execution_identity.input_snapshots"
    )
    for replica_id, raw_record in input_snapshots.items():
        record = _require_mapping(
            raw_record, f"execution_identity.input_snapshots.{replica_id}"
        )
        replica_root = input_root / "replicas" / str(replica_id)
        snapshot = replica_root / "source_snapshot.data"
        metadata = replica_root / "snapshot_contract.json"
        if (
            not snapshot.is_file()
            or snapshot.stat().st_size != int(record.get("snapshot_bytes", -1))
            or sha256_file(snapshot) != record.get("snapshot_sha256")
        ):
            raise StageIntegrityError(
                f"snapshotted input structure changed before stage launch: {replica_id}"
            )
        if (
            not metadata.is_file()
            or metadata.stat().st_size != int(record.get("metadata_bytes", -1))
            or sha256_file(metadata) != record.get("metadata_sha256")
        ):
            raise StageIntegrityError(
                f"snapshotted input contract changed before stage launch: {replica_id}"
            )
    engine_identity = _require_mapping(
        identity.get("engine"), "execution_identity.engine"
    )
    configured_engine = _require_mapping(
        resolved.get("engine"), "resolved.engine"
    )
    configured_command = list(
        _require_sequence(
            configured_engine.get("lammps_command"), "engine.lammps_command"
        )
    )
    if configured_command != engine_identity.get("command_argv"):
        raise StageIntegrityError("pinned LAMMPS command argv changed")
    if configured_engine.get("device") != engine_identity.get("device"):
        raise StageIntegrityError("pinned MACE device changed")
    command_files = engine_identity.get("command_files")
    if not isinstance(command_files, list):
        raise StageIntegrityError("execution identity lacks command-file records")
    for record in command_files:
        if not isinstance(record, Mapping):
            raise StageIntegrityError("invalid command-file identity record")
        executable = Path(str(record.get("resolved_path", "")))
        if (
            not executable.is_file()
            or executable.stat().st_size != int(record.get("bytes", -1))
            or sha256_file(executable) != record.get("sha256")
        ):
            raise StageIntegrityError(
                f"pinned command file changed or disappeared: {executable}"
            )
    configured_dependencies = list(
        _require_sequence(
            configured_engine.get("runtime_dependencies"),
            "engine.runtime_dependencies",
        )
    )
    if configured_dependencies != engine_identity.get(
        "declared_runtime_dependencies"
    ):
        raise StageIntegrityError("pinned engine runtime dependencies changed")
    dependency_files = engine_identity.get("runtime_dependency_files")
    if not isinstance(dependency_files, list) or len(dependency_files) != len(
        configured_dependencies
    ):
        raise StageIntegrityError(
            "execution identity lacks verified runtime dependency records"
        )
    for record in dependency_files:
        if not isinstance(record, Mapping):
            raise StageIntegrityError("invalid runtime dependency identity record")
        dependency = Path(str(record.get("resolved_path", "")))
        if (
            not dependency.is_file()
            or dependency.stat().st_size != int(record.get("bytes", -1))
            or sha256_file(dependency) != record.get("sha256")
        ):
            raise StageIntegrityError(
                f"pinned engine runtime dependency changed: {dependency}"
            )


def build_density_reuse_key(
    resolved: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    replica_id: str,
    seed: int,
    runtime_input_sha256: Mapping[str, Any],
    engine_identity: Mapping[str, Any] | None = None,
    step_plan: Mapping[str, Any] | None = None,
) -> str:
    """Build the plateau-local key shared by Tg and density-only modes."""

    if state.get("use_for_density") is not True:
        raise ThermalSimulationError(
            "only an explicitly tagged density target can have a density reuse key"
        )
    engine = _require_mapping(resolved.get("engine"), "resolved.engine")
    npt = _require_mapping(resolved.get("npt"), "resolved.npt")
    system = _require_mapping(resolved.get("system"), "resolved.system")
    if engine_identity is None:
        execution_identity = resolved.get("execution_identity")
        if isinstance(execution_identity, Mapping) and isinstance(
            execution_identity.get("engine"), Mapping
        ):
            engine_identity = execution_identity["engine"]
        else:
            engine_identity = build_engine_identity(
                resolved, require_primary_executable=False
            )
    if step_plan is None:
        equilibration_steps = ps_to_steps(
            npt["equilibration_ps"], engine["dt_ps"]
        )
        production_steps = ps_to_steps(npt["production_ps"], engine["dt_ps"])
        ramp_duration = _decimal(npt["ramp_ps"], "npt.ramp_ps")
        ramp_steps = (
            ps_to_steps(ramp_duration, engine["dt_ps"])
            if ramp_duration > 0
            else 0
        )
    else:
        equilibration_steps = int(step_plan["constant_equilibration_steps"])
        production_steps = int(step_plan["production_steps"])
        ramp_steps = int(step_plan["ramp_steps"])
    return canonical_sha256(
        {
            "schema_version": "thermal-properties-density-reuse-key/v1",
            "target_temperature_K": state["temperature_start_k"],
            "target_pressure_bar": state["pressure_start_bar"],
            "ensemble": "npt",
            "pressure_coupling": "iso",
            "timestep_ps": engine["dt_ps"],
            "thermostat": "nose-hoover",
            "barostat": "nose-hoover",
            "tdamp_ps": engine["tdamp_ps"],
            "pdamp_ps": engine["pdamp_ps"],
            "ramp_steps": ramp_steps,
            "equilibration_steps": equilibration_steps,
            "production_steps": production_steps,
            "sample_every_steps": ps_to_steps(
                npt["sample_interval_ps"], engine["dt_ps"]
            ),
            "thermo_every_steps": ps_to_steps(
                npt["thermo_interval_ps"], engine["dt_ps"]
            ),
            "mace_elements": list(system["mace_elements"]),
            "replica_id": replica_id,
            "seed": seed,
            "runtime_input_sha256": dict(runtime_input_sha256),
            "engine_identity": dict(engine_identity),
        }
    )


def _serialize_reuse_decision(decision: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action": decision.action.value,
        "reason": decision.reason,
    }
    if decision.candidate is not None:
        candidate = decision.candidate
        payload["candidate"] = {
            "run_id": candidate.provenance.run_id,
            "replica_id": candidate.provenance.replica_id,
            "state_point_id": candidate.provenance.state_point_id,
            "state_point_provenance": str(candidate.provenance_path.resolve()),
            "run_manifest": str(candidate.run_manifest_path),
            "run_manifest_sha256": sha256_file(candidate.run_manifest_path),
            "production_samples": str(candidate.artifact_path.resolve()),
            "production_samples_sha256": candidate.artifact.sha256,
        }
    return payload


def plan_density_reuse(resolved: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the v1 zero/one/multiple density-reuse decision without writes.

    Search is opt-in through ``reuse.search_roots`` or an explicit selected
    state-point provenance path.  V1 intentionally accepts a single replica
    for cross-run reuse so it never aggregates independent trajectories or
    mixes reused and newly simulated replicas implicitly.
    """

    if resolved.get("thermal_mode") != "density":
        return {"action": "not_applicable", "reason": "Tg mode owns its schedule"}
    if (
        resolved.get("run_class") != "PRODUCTION"
        or resolved.get("scientific_eligible") is not True
        or resolved.get("reuse_eligible") is not True
    ):
        return {
            "action": ReuseAction.RUN_NEW.value,
            "reason": (
                "automatic cross-run reuse is disabled by run_class and "
                "eligibility gates"
            ),
        }
    raw_reuse = resolved.get("reuse")
    if not isinstance(raw_reuse, Mapping):
        return {"action": ReuseAction.RUN_NEW.value, "reason": "reuse not configured"}
    search_roots = raw_reuse.get("search_roots")
    selected = raw_reuse.get("selected_state_point_provenance")
    if not isinstance(search_roots, Sequence) or isinstance(search_roots, (str, bytes)):
        raise ThermalSimulationError("reuse.search_roots must be an array")
    if not search_roots and selected is None:
        return {
            "action": ReuseAction.RUN_NEW.value,
            "reason": "reuse search is explicitly disabled",
        }
    replicas = resolved.get("replicas")
    if not isinstance(replicas, list) or len(replicas) != 1:
        raise ThermalSimulationError(
            "cross-run density reuse v1 requires exactly one replica; split "
            "multi-replica requests and analyze them explicitly"
        )
    replica = _require_mapping(replicas[0], "resolved.replicas[0]")
    replica_id = str(replica.get("replica_id", ""))
    seed = int(replica.get("seed"))
    plan = _require_mapping(resolved.get("plan"), "resolved.plan")
    steps = plan.get("steps")
    if not isinstance(steps, list):
        raise ThermalSimulationError("resolved plan has no state-point steps")
    targets = [
        item
        for item in steps
        if isinstance(item, Mapping)
        and item.get("replica_id") == replica_id
        and isinstance(item.get("state"), Mapping)
        and item["state"].get("use_for_density") is True
    ]
    if len(targets) != 1:
        raise ThermalSimulationError(
            "density reuse requires exactly one explicitly tagged density target"
        )
    execution_identity = build_execution_identity(resolved)
    replica_snapshot_identity = _require_mapping(
        _require_mapping(
            execution_identity.get("input_snapshots"),
            "execution_identity.input_snapshots",
        ).get(replica_id),
        f"execution_identity.input_snapshots.{replica_id}",
    )
    state = targets[0]["state"]
    density_reuse_key = build_density_reuse_key(
        resolved,
        state,
        replica_id=replica_id,
        seed=seed,
        runtime_input_sha256=execution_identity["runtime_input_sha256"],
        engine_identity=execution_identity["engine"],
        step_plan=targets[0],
    )
    density_qc_policy = _density_qc_policy_identity()
    density_qc_code_sha256 = density_qc_implementation_sha256(
        execution_identity
    )
    request = ReuseRequest(
        replica_id=replica_id,
        run_class=str(resolved["run_class"]),
        scientific_eligible=bool(resolved["scientific_eligible"]),
        reuse_eligible=bool(resolved["reuse_eligible"]),
        density_reuse_key=density_reuse_key,
        density_qc_policy_id=density_qc_policy["policy_id"],
        density_qc_policy_sha256=density_qc_policy["policy_sha256"],
        density_qc_implementation_sha256=density_qc_code_sha256,
        input_snapshot_sha256=str(replica_snapshot_identity["snapshot_sha256"]),
        model_sha256=execution_identity["model"]["sha256"],
        artifact_name="production_samples",
        require_analysis_success=False,
    )
    try:
        if selected is not None:
            selected_path = Path(str(selected)).expanduser().resolve()
            record = read_state_point_provenance(selected_path)
            decision = select_reuse_candidate([(selected_path, record)], request)
            if decision.action is not ReuseAction.REUSE:
                raise ThermalSimulationError(
                    "the explicitly selected density source is not eligible"
                )
        else:
            decision = resolve_cross_run_reuse(
                [Path(str(root)).expanduser().resolve() for root in search_roots],
                request,
            )
    except (AmbiguousReuseError, ReuseError, ProvenanceError) as exc:
        raise ThermalSimulationError(f"density reuse resolution failed: {exc}") from exc
    payload = _serialize_reuse_decision(decision)
    payload["request"] = {
        "replica_id": replica_id,
        "run_class": request.run_class,
        "scientific_eligible": request.scientific_eligible,
        "reuse_eligible": request.reuse_eligible,
        "density_reuse_key": density_reuse_key,
        "density_qc_policy_id": request.density_qc_policy_id,
        "density_qc_policy_sha256": request.density_qc_policy_sha256,
        "density_qc_implementation_sha256": (
            request.density_qc_implementation_sha256
        ),
        "input_snapshot_sha256": request.input_snapshot_sha256,
        "model_sha256": request.model_sha256,
        "artifact_name": request.artifact_name,
    }
    return payload


def _claim_path_for_run(root: Path, run_id: str) -> Path:
    return root / CLAIM_ROOT_NAME / f"{run_id}.claim"


def _acquire_direct_run_owner(root: Path, run_id: str) -> tuple[Path, int, str]:
    owner_root = root / DIRECT_OWNER_ROOT_NAME
    if os.path.lexists(owner_root):
        if owner_root.is_symlink() or not owner_root.is_dir():
            raise ThermalSimulationError(
                f"direct-run owner root is unsafe: {owner_root}"
            )
    else:
        try:
            owner_root.mkdir(mode=0o700)
        except FileExistsError:
            if owner_root.is_symlink() or not owner_root.is_dir():
                raise ThermalSimulationError(
                    f"direct-run owner root raced with an unsafe path: {owner_root}"
                )
    owner_path = owner_root / f"{run_id}.lock"
    if os.path.lexists(owner_path) and owner_path.is_symlink():
        raise ThermalSimulationError(f"direct-run owner lock is a symlink: {owner_path}")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(owner_path, flags, 0o600)
    except OSError as exc:
        raise ThermalSimulationError(
            f"cannot open direct-run owner lock: {owner_path}"
        ) from exc
    owner_uuid = str(uuid.uuid4())
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ThermalSimulationError("direct-run owner lock is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise ThermalSimulationError(
                    f"another direct process owns thermal run {run_id}: {owner_path}"
                ) from exc
            raise
        record = (
            json.dumps(
                {
                    "schema_version": "thermal-properties-direct-run-owner/v1",
                    "run_id": run_id,
                    "canonical_output_root": str(root),
                    "owner_uuid": owner_uuid,
                    "pid": os.getpid(),
                    "active": True,
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, record)
        os.fsync(descriptor)
    except BaseException:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
        raise
    return owner_path, descriptor, owner_uuid


def _release_direct_run_owner(
    owner_path: Path, descriptor: int, owner_uuid: str
) -> None:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            record = json.loads(os.read(descriptor, 65536).decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ThermalSimulationError(
                f"cannot verify direct-run owner record: {owner_path}"
            ) from exc
        if record.get("owner_uuid") != owner_uuid:
            raise ThermalSimulationError("direct-run owner UUID changed unexpectedly")
        record["active"] = False
        record["released_by_pid"] = os.getpid()
        encoded = (
            json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def execute_thermal_campaign(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    run_id: str | None = None,
    process_runner: ProcessRunner | None = None,
    qc_evaluator_factory: Callable[
        [Mapping[str, Any]], Callable[[Path, Path], str | Mapping[str, Any]] | None
    ]
    | None = None,
    output_ownership: OutputOwnership | None = None,
) -> Path:
    """Execute with one atomic direct owner or an authenticated claim owner."""

    if output_ownership is not None:
        return _execute_thermal_campaign_impl(
            config_path,
            output_root=output_root,
            run_id=run_id,
            process_runner=process_runner,
            qc_evaluator_factory=qc_evaluator_factory,
            output_ownership=output_ownership,
        )

    source_config = Path(config_path).resolve()
    resolved = resolve_thermal_config(source_config)
    reuse_decision = plan_density_reuse(resolved)
    if reuse_decision.get("action") == ReuseAction.REUSE.value:
        return _execute_thermal_campaign_impl(
            source_config,
            output_root=output_root,
            run_id=run_id,
            process_runner=process_runner,
            qc_evaluator_factory=qc_evaluator_factory,
            output_ownership=None,
            pre_resolved=resolved,
            preplanned_reuse_decision=reuse_decision,
        )
    configured_root = output_root if output_root is not None else resolved.get("output_root")
    if not isinstance(configured_root, (str, Path)) or not str(configured_root):
        raise ThermalSimulationError("an output_root is required")
    identifier = _validate_run_id(run_id or f"run_{uuid.uuid4()}")
    root = Path(configured_root).resolve()
    _guard_density_parent_destination(resolved, root, identifier)
    root.mkdir(parents=True, exist_ok=True)
    final = root / identifier
    incomplete = root / f"{identifier}.incomplete"
    claim_dir = _claim_path_for_run(root, identifier)
    if os.path.lexists(final):
        raise ThermalSimulationError(f"final run already exists: {final}")
    if os.path.lexists(claim_dir):
        raise ThermalSimulationError(
            f"claim-bound run requires authenticated output ownership: {claim_dir}"
        )
    if os.path.lexists(incomplete / OWNER_MARKER_NAME):
        raise ThermalSimulationError(
            f"claimed incomplete run cannot be entered directly: {incomplete}"
        )
    owner_path, owner_descriptor, owner_uuid = _acquire_direct_run_owner(
        root, identifier
    )
    created_incomplete = False
    try:
        if os.path.lexists(incomplete):
            if incomplete.is_symlink() or not incomplete.is_dir():
                raise ThermalSimulationError(
                    f"resume target is not a safe directory: {incomplete}"
                )
            if os.path.lexists(incomplete / OWNER_MARKER_NAME):
                raise ThermalSimulationError(
                    f"claimed incomplete run cannot be entered directly: {incomplete}"
                )
        else:
            incomplete.mkdir(parents=False, exist_ok=False)
            created_incomplete = True
        if os.path.lexists(claim_dir):
            if created_incomplete:
                try:
                    incomplete.rmdir()
                except OSError:
                    pass
            raise ThermalSimulationError(
                f"a scheduler claim raced with direct execution: {claim_dir}"
            )
        return _execute_thermal_campaign_impl(
            source_config,
            output_root=root,
            run_id=identifier,
            process_runner=process_runner,
            qc_evaluator_factory=qc_evaluator_factory,
            output_ownership=None,
            pre_resolved=resolved,
            preplanned_reuse_decision=reuse_decision,
        )
    finally:
        _release_direct_run_owner(owner_path, owner_descriptor, owner_uuid)


def _execute_thermal_campaign_impl(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    run_id: str | None = None,
    process_runner: ProcessRunner | None = None,
    qc_evaluator_factory: Callable[
        [Mapping[str, Any]], Callable[[Path, Path], str | Mapping[str, Any]] | None
    ]
    | None = None,
    output_ownership: OutputOwnership | None = None,
    pre_resolved: dict[str, Any] | None = None,
    preplanned_reuse_decision: Mapping[str, Any] | None = None,
) -> Path:
    """Execute a resolved thermal campaign with per-stage idempotent resume.

    This function is the sole expensive-MD orchestration path.  Both modes call
    the same initializer and NPT state-point input.  It never launches an
    analysis or an automatic continuation.  Tests inject a synthetic runner;
    real execution is intentionally left to an explicit CLI ``run`` command.
    """

    source_config = Path(config_path).resolve()
    resolved = (
        dict(pre_resolved)
        if pre_resolved is not None
        else resolve_thermal_config(source_config)
    )
    configured_root = output_root if output_root is not None else resolved.get("output_root")
    if not isinstance(configured_root, (str, Path)) or not str(configured_root):
        raise ThermalSimulationError("an output_root is required")
    reuse_decision = (
        dict(preplanned_reuse_decision)
        if preplanned_reuse_decision is not None
        else plan_density_reuse(resolved)
    )
    if reuse_decision.get("action") == ReuseAction.REUSE.value:
        if output_ownership is not None:
            raise ThermalSimulationError(
                "claim-bound scheduler execution cannot switch to a newly "
                "available reuse candidate after submission"
            )
        candidate = reuse_decision.get("candidate")
        if not isinstance(candidate, Mapping) or not isinstance(
            candidate.get("run_manifest"), str
        ):
            raise ThermalSimulationError("reuse decision lacks a run manifest")
        return Path(candidate["run_manifest"])
    execution_identity = build_execution_identity(
        resolved,
        require_primary_executable=process_runner is None,
    )
    resolved["execution_identity"] = execution_identity
    resolved["resolved_spec_sha256"] = resolved_spec_sha256(resolved)
    identifier = _validate_run_id(run_id or f"run_{uuid.uuid4()}")
    root = Path(configured_root).resolve()
    _guard_density_parent_destination(resolved, root, identifier)
    incomplete = root / f"{identifier}.incomplete"
    final = root / identifier
    if os.path.lexists(final):
        raise ThermalSimulationError(f"final run already exists: {final}")
    if output_ownership is not None:
        expected_owner_values = {
            "run_root": root,
            "run_id": identifier,
            "expected_output_path": final,
            "incomplete_output_path": incomplete,
            "execution_resolved_spec_sha256": resolved["resolved_spec_sha256"],
        }
        for field, expected_value in expected_owner_values.items():
            if getattr(output_ownership, field) != expected_value:
                raise ThermalSimulationError(
                    f"submission output ownership mismatch: {field}"
                )
        if not incomplete.is_dir() or incomplete.is_symlink():
            raise ThermalSimulationError(
                f"claimed incomplete run directory is invalid: {incomplete}"
            )
        owner_marker = incomplete / OWNER_MARKER_NAME
        if not owner_marker.is_file() or owner_marker.is_symlink():
            raise ThermalSimulationError(
                f"claimed output owner marker is missing: {owner_marker}"
            )
        try:
            owner_payload = json.loads(owner_marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ThermalSimulationError(
                f"claimed output owner marker is invalid: {owner_marker}"
            ) from exc
        expected_marker = {
            "claim_uuid": output_ownership.claim_uuid,
            "canonical_run_id": identifier,
            "execution_resolved_spec_sha256": resolved["resolved_spec_sha256"],
            "attempt_uuid": output_ownership.attempt_uuid,
            "owner_uuid": output_ownership.owner_uuid,
        }
        for field, expected_value in expected_marker.items():
            if owner_payload.get(field) != expected_value:
                raise ThermalSimulationError(
                    f"claimed output owner marker mismatch: {field}"
                )
        if not output_ownership.resume:
            unexpected = [
                path.name for path in incomplete.iterdir() if path.name != OWNER_MARKER_NAME
            ]
            if unexpected:
                raise ThermalSimulationError(
                    "initial claimed output is not empty before campaign setup"
                )
    elif os.path.lexists(incomplete):
        if not incomplete.is_dir() or incomplete.is_symlink():
            raise ThermalSimulationError(f"resume target is not a directory: {incomplete}")
    else:
        incomplete.mkdir(parents=True, exist_ok=False)

    input_root = incomplete / "input"
    input_root.mkdir(parents=True, exist_ok=True)
    parent_identity = execution_identity.get("density_parent_restart")
    parent_input_records: list[dict[str, Any]] = []
    if parent_identity is not None:
        for key, record in parent_identity["source_files"].items():
            destination = _density_parent_copy_path(input_root, key)
            copied = _copy_or_verify_runtime_input(record["path"], destination)
            if copied["copied_sha256"] != record["sha256"]:
                raise StageIntegrityError(f"parent evidence changed during intake: {key}")
            parent_input_records.append({
                "role": key,
                "source_path": record["path"],
                "source_sha256": record["sha256"],
                "artifact": artifact_record(destination, relative_to=incomplete),
            })
    system = _require_mapping(resolved["system"], "resolved.system")
    identity_snapshots = _require_mapping(
        execution_identity.get("input_snapshots"),
        "execution_identity.input_snapshots",
    )
    replica_inputs: dict[str, dict[str, Any]] = {}
    for raw_replica in resolved["replicas"]:
        replica = _require_mapping(raw_replica, "resolved replica")
        replica_id = str(replica["replica_id"])
        snapshot_identity = _require_mapping(
            identity_snapshots.get(replica_id),
            f"execution_identity.input_snapshots.{replica_id}",
        )
        replica_input_root = input_root / "replicas" / replica_id
        snapshot_copy = _copy_or_verify_runtime_input(
            str(replica["input_data"]),
            replica_input_root / "source_snapshot.data",
        )
        metadata_copy = _copy_or_verify_runtime_input(
            str(replica["snapshot_metadata"]),
            replica_input_root / "snapshot_contract.json",
        )
        if (
            snapshot_copy["source_sha256"]
            != snapshot_identity.get("snapshot_sha256")
            or metadata_copy["source_sha256"]
            != snapshot_identity.get("metadata_sha256")
        ):
            raise StageIntegrityError(
                f"replica snapshot contract changed during campaign setup: {replica_id}"
            )
        replica_inputs[replica_id] = {
            "snapshot": snapshot_copy,
            "metadata": metadata_copy,
            "identity": dict(snapshot_identity),
            "copied_snapshot_path": replica_input_root / "source_snapshot.data",
        }
    package_root = Path(__file__).resolve().parent
    snapshotted_inputs: list[dict[str, Any]] = []
    for name in RUNTIME_INPUT_FILES:
        snapshotted_inputs.append(
            _copy_or_verify_runtime_input(
                package_root / "lammps" / name, input_root / name
            )
        )
    copied_runtime_hashes = {
        Path(item["copied_path"]).name: item["copied_sha256"]
        for item in snapshotted_inputs
    }
    if copied_runtime_hashes != execution_identity["runtime_input_sha256"]:
        raise StageIntegrityError("thermal runtime inputs changed during campaign setup")
    model_identity = dict(execution_identity["model"])
    spec_path = incomplete / "spec" / "run_spec.json"
    if spec_path.exists():
        existing = _read_json(spec_path)
        embedded_digest = existing.get("resolved_spec_sha256")
        if embedded_digest != resolved_spec_sha256(existing):
            raise StageIntegrityError(
                "stored resolved run specification digest is invalid"
            )
        if embedded_digest != resolved["resolved_spec_sha256"]:
            raise StageIntegrityError("resolved run specification changed during resume")
    else:
        _atomic_write_json(spec_path, resolved)
    engine = _require_mapping(resolved["engine"], "resolved.engine")
    lammps_command = _require_sequence(
        execution_identity["engine"].get("launch_argv"),
        "execution_identity.engine.launch_argv",
    )
    if not lammps_command or not all(isinstance(item, str) and item for item in lammps_command):
        raise ThermalSimulationError("engine.lammps_command must contain command arguments")
    initialize_command = tuple(lammps_command) + (
        "-in",
        str((input_root / "in.initialize_mace_mh1.lmp").resolve()),
    )
    npt_command = tuple(lammps_command) + (
        "-in",
        str((input_root / "in.npt_stage_mace_mh1.lmp").resolve()),
    )
    initialize = _require_mapping(resolved["initialize"], "resolved.initialize")
    npt = _require_mapping(resolved["npt"], "resolved.npt")
    init_steps = ps_to_steps(initialize["nvt_ps"], engine["dt_ps"])
    transition_steps = ps_to_steps(
        initialize["mace_transition_npt_ps"], engine["dt_ps"]
    )
    sample_every_steps = ps_to_steps(
        npt["sample_interval_ps"], engine["dt_ps"]
    )
    sampling_histories: dict[str, list[dict[str, Any]]] = {}

    def sampling_evaluator(spec):
        if qc_evaluator_factory is None:
            return _default_statepoint_qc_evaluator(resolved, spec)
        injected = qc_evaluator_factory(spec)
        if injected is None:
            raise ThermalSimulationError("qc_evaluator_factory returned no evaluator")
        return _bind_statepoint_qc_evaluator(
            injected, _statepoint_qc_policy_requirements(spec),
            qc_implementation_sha256=density_qc_implementation_sha256(resolved["execution_identity"]),
        )
    thermo_every_steps = ps_to_steps(
        npt["thermo_interval_ps"], engine["dt_ps"]
    )

    resolved_steps = resolved["plan"]["steps"]
    state_results: list[StageExecutionResult] = []
    initialization_results: list[StageExecutionResult] = []
    initialization_records: list[dict[str, Any]] = []
    state_records: list[dict[str, Any]] = []
    history_by_replica: dict[str, list[str]] = {}
    for replica in resolved["replicas"]:
        replica_id = str(replica["replica_id"])
        seed = int(replica["seed"])
        if parent_identity is not None:
            initialize_restart = _density_parent_copy_path(input_root, "restart")
            transition_start_step = int(parent_identity["start_step"])
        else:
            init_dir = incomplete / "replicas" / replica_id / "initialize"
            init_environment = _campaign_stage_environment(
                resolved,
                input_data=replica_inputs[replica_id]["copied_snapshot_path"],
                stage_dir=init_dir,
            )
            init_environment["THERMAL_SEED"] = str(seed)
            init_spec = {
                "stage_id": f"{replica_id}__initialize",
                "stage_role": "initialization",
                "replica_id": replica_id,
                "seed": seed,
                "start_step": 0,
                "equilibration_steps": 0,
                "production_steps": init_steps,
                "sample_every_steps": sample_every_steps,
                "initialize_velocities": True,
                "predecessor_restart": None,
                "working_directory": os.path.relpath(input_root, start=init_dir),
                "run_class": resolved["run_class"],
                "scientific_eligible": resolved["scientific_eligible"],
                "reuse_eligible": resolved["reuse_eligible"],
                "execution_identity": resolved["execution_identity"],
            }
            _verify_execution_identity(resolved, input_root)
            initialization_result = execute_npt_stage(
                stage_dir=init_dir,
                stage_spec=init_spec,
                command=initialize_command,
                environment=init_environment,
                process_runner=process_runner,
            )
            initialization_results.append(initialization_result)
            initialization_records.append(
                {
                    "stage_id": initialization_result.stage_id,
                    "replica_id": replica_id,
                    "qc_status": initialization_result.qc_status,
                    "stage_spec_sha256": canonical_sha256(
                        _read_json(init_dir / "stage_spec.json")
                    ),
                    "stage_manifest": artifact_record(
                        initialization_result.manifest_path, relative_to=incomplete
                    ),
                }
            )
            initialize_restart = init_dir / "restart.final"
            transition_start_step = init_steps

        transition_dir = (
            incomplete / "replicas" / replica_id / "mace_transition_npt"
        )
        transition_environment = _campaign_stage_environment(
            resolved,
            input_data=replica_inputs[replica_id]["copied_snapshot_path"],
            stage_dir=transition_dir,
        )
        transition_temperature = float(initialize["temperature_k"])
        transition_pressure = float(initialize["mace_transition_pressure_bar"])
        transition_environment.update(
            {
                "THERMAL_TARGET_TEMP_K": str(transition_temperature),
                "THERMAL_TARGET_PRESS_BAR": str(transition_pressure),
            }
        )
        transition_state = StatePointConfig.constant(
            state_id="mace_transition_npt",
            temperature_k=transition_temperature,
            pressure_bar=transition_pressure,
            duration_ps=initialize["mace_transition_npt_ps"],
            stage_kind=StageKind.EQUILIBRATION,
            use_for_tg_fit=False,
            use_for_density=False,
        ).to_dict()
        transition_spec = {
            "stage_id": f"{replica_id}__mace_transition_npt",
            "stage_role": "mace_transition_npt",
            "replica_id": replica_id,
            "seed": seed,
            "start_step": transition_start_step,
            "state": transition_state,
            "roles": ["mace_transition"],
            "branch": "mace_transition",
            "ramp_steps": 0,
            "constant_equilibration_steps": 0,
            "equilibration_steps": 0,
            "production_steps": transition_steps,
            "temperature_ramp_start_k": transition_temperature,
            "temperature_ramp_end_k": transition_temperature,
            "sample_every_steps": sample_every_steps,
            "initialize_velocities": False,
            "predecessor_restart": os.path.relpath(
                initialize_restart, start=transition_dir
            ),
            "working_directory": os.path.relpath(input_root, start=transition_dir),
            "run_class": resolved["run_class"],
            "scientific_eligible": resolved["scientific_eligible"],
            "reuse_eligible": False,
            "execution_identity": resolved["execution_identity"],
        }
        transition_requirements = _statepoint_qc_policy_requirements(
            transition_spec
        )
        if qc_evaluator_factory is not None:
            transition_injected = qc_evaluator_factory(transition_spec)
            if transition_injected is None:
                raise ThermalSimulationError(
                    "qc_evaluator_factory returned no evaluator"
                )
            transition_evaluator = _bind_statepoint_qc_evaluator(
                transition_injected,
                transition_requirements,
                qc_implementation_sha256=density_qc_implementation_sha256(
                    resolved["execution_identity"]
                ),
            )
        else:
            transition_evaluator = _default_statepoint_qc_evaluator(
                resolved, transition_spec
            )
        _verify_execution_identity(resolved, input_root)
        if resolved.get("sampling_continuation") is not None:
            from .sampling_continuation import execute_sampling_stage

            transition_result, transition_spec, transition_history = execute_sampling_stage(
                resolved=resolved, run_root=incomplete, stage_dir=transition_dir,
                stage_spec=transition_spec, command=npt_command, environment=transition_environment,
                process_runner=process_runner, evaluator_factory=sampling_evaluator, input_root=input_root,
            )
            transition_dir = transition_result.stage_dir
            sampling_histories[transition_result.stage_id] = transition_history
        else:
            transition_result = execute_npt_stage(
                stage_dir=transition_dir, stage_spec=transition_spec, command=npt_command,
                environment=transition_environment, process_runner=process_runner,
                qc_evaluator=transition_evaluator,
            )
        initialization_results.append(transition_result)
        transition_qc_passed = transition_result.qc_status == "PASS"
        transition_qc_nonblocking = (
            not transition_qc_passed
            and resolved["run_class"] == "ENGINEERING_SMOKE"
            and resolved["scientific_eligible"] is False
            and resolved["reuse_eligible"] is False
        )
        transition_qc_gate = {
            "qc_status": transition_result.qc_status,
            "decision": (
                "PASSED"
                if transition_qc_passed
                else (
                    "RECORDED_NONBLOCKING_ENGINEERING_SMOKE"
                    if transition_qc_nonblocking
                    else "BLOCKED"
                )
            ),
            "property_branch_started": (
                transition_qc_passed or transition_qc_nonblocking
            ),
            "run_class": resolved["run_class"],
            "scientific_eligible": resolved["scientific_eligible"],
            "reuse_eligible": resolved["reuse_eligible"],
        }
        initialization_records.append(
            {
                "stage_id": transition_result.stage_id,
                "stage_role": "mace_transition_npt",
                "replica_id": replica_id,
                "qc_status": transition_result.qc_status,
                "qc_gate": transition_qc_gate,
                "stage_spec_sha256": canonical_sha256(
                    _read_json(transition_dir / "stage_spec.json")
                ),
                "stage_manifest": artifact_record(
                    transition_result.manifest_path, relative_to=incomplete
                ),
            }
        )
        if not transition_qc_passed and not transition_qc_nonblocking:
            raise ThermalSimulationError(
                "MACE transition NPT did not pass stationarity/structural QC; "
                "the property branch was not started"
            )
        predecessor = transition_dir / "restart.final"
        next_stage_start_step = int(transition_spec["start_step"]) + int(transition_spec["production_steps"])
        history_by_replica[replica_id] = [
            sha256_file(initialize_restart),
            sha256_file(predecessor),
        ]

        replica_steps = [
            item for item in resolved_steps if item["replica_id"] == replica_id
        ]
        for item in replica_steps:
            state = item["state"]
            state_ramp_steps = int(item["ramp_steps"])
            state_equil_steps = int(item["constant_equilibration_steps"])
            state_preproduction_steps = int(item["preproduction_steps"])
            state_prod_steps = int(item["production_steps"])
            sequence_index = int(item["sequence_index"])
            physical_state_id = str(state["state_id"])
            analysis_state_id = f"{replica_id}__{physical_state_id}"
            stage_dir = (
                incomplete
                / "replicas"
                / replica_id
                / "stages"
                / f"{sequence_index:04d}_{physical_state_id}"
            )
            stage_environment = _campaign_stage_environment(
                resolved,
                input_data=replica_inputs[replica_id]["copied_snapshot_path"],
                stage_dir=stage_dir,
            )
            stage_environment.update(
                {
                    "THERMAL_TARGET_TEMP_K": str(state["temperature_start_k"]),
                    "THERMAL_TARGET_PRESS_BAR": str(state["pressure_start_bar"]),
                }
            )
            history_key = canonical_sha256(history_by_replica[replica_id])
            density_reuse_key = None
            if state.get("use_for_density") is True:
                density_reuse_key = build_density_reuse_key(
                    resolved,
                    state,
                    replica_id=replica_id,
                    seed=seed,
                    runtime_input_sha256=resolved["execution_identity"][
                        "runtime_input_sha256"
                    ],
                    engine_identity=resolved["execution_identity"]["engine"],
                    step_plan=item,
                )
            trajectory_key = canonical_sha256(
                {
                    "schema_version": "thermal-properties-trajectory-key/v1",
                    "state": state,
                    "history_key": history_key,
                    "protocol_sha256": resolved["protocol_sha256"],
                    "sequence_index": sequence_index,
                    "execution_identity": resolved["execution_identity"],
                }
            )
            stage_spec = {
                "stage_id": analysis_state_id,
                "physical_state_id": physical_state_id,
                "stage_role": "npt_state_point",
                "replica_id": replica_id,
                "sequence_index": sequence_index,
                "seed": seed,
                "start_step": next_stage_start_step,
                "state": state,
                "roles": list(item["roles"]),
                "branch": "cooling" if resolved["thermal_mode"] == "tg" else "hold",
                "ramp_steps": state_ramp_steps,
                "constant_equilibration_steps": state_equil_steps,
                "equilibration_steps": state_preproduction_steps,
                "production_steps": state_prod_steps,
                "temperature_ramp_start_k": item[
                    "temperature_ramp_start_k"
                ],
                "temperature_ramp_end_k": item["temperature_ramp_end_k"],
                "sample_every_steps": sample_every_steps,
                "initialize_velocities": False,
                "predecessor_restart": os.path.relpath(predecessor, start=stage_dir),
                "working_directory": os.path.relpath(input_root, start=stage_dir),
                "history_key": history_key,
                "trajectory_key": trajectory_key,
                "density_reuse_key": density_reuse_key,
                "run_class": resolved["run_class"],
                "scientific_eligible": resolved["scientific_eligible"],
                "reuse_eligible": resolved["reuse_eligible"],
                "execution_identity": resolved["execution_identity"],
            }
            _verify_execution_identity(resolved, input_root)
            requirements = _statepoint_qc_policy_requirements(stage_spec)
            if qc_evaluator_factory is not None:
                injected_evaluator = qc_evaluator_factory(stage_spec)
                if injected_evaluator is None:
                    raise ThermalSimulationError(
                        "qc_evaluator_factory returned no evaluator"
                    )
                evaluator = _bind_statepoint_qc_evaluator(
                    injected_evaluator,
                    requirements,
                    qc_implementation_sha256=density_qc_implementation_sha256(
                        resolved["execution_identity"]
                    ),
                )
            else:
                evaluator = _default_statepoint_qc_evaluator(resolved, stage_spec)
            if resolved.get("sampling_continuation") is not None:
                from .sampling_continuation import execute_sampling_stage

                result, stage_spec, state_history = execute_sampling_stage(
                    resolved=resolved, run_root=incomplete, stage_dir=stage_dir,
                    stage_spec=stage_spec, command=npt_command, environment=stage_environment,
                    process_runner=process_runner, evaluator_factory=sampling_evaluator, input_root=input_root,
                )
                stage_dir = result.stage_dir
                sampling_histories[result.stage_id] = state_history
                state_ramp_steps = int(stage_spec["ramp_steps"])
                state_equil_steps = int(stage_spec["constant_equilibration_steps"])
                state_preproduction_steps = int(stage_spec["equilibration_steps"])
                state_prod_steps = int(stage_spec["production_steps"])
            else:
                result = execute_npt_stage(
                    stage_dir=stage_dir, stage_spec=stage_spec, command=npt_command,
                    environment=stage_environment, process_runner=process_runner, qc_evaluator=evaluator,
                )
            state_results.append(result)
            production = stage_dir / "samples" / (
                "production.cumulative.csv" if stage_spec.get("sampling_production_sources") else "production.csv"
            )
            production_artifact = artifact_record(production, relative_to=incomplete)
            production_artifact.update(
                {"phase": "production", "role": "thermo_samples"}
            )
            qc_path = stage_dir / "qc" / "state_point_qc.json"
            qc_payload = _read_json(qc_path)
            qc_policy_results = _policy_result_summary(qc_payload)
            qc_implementation_sha256 = str(
                qc_payload.get("qc_implementation_sha256", "")
            )
            expected_qc_implementation = density_qc_implementation_sha256(
                resolved["execution_identity"]
            )
            if qc_implementation_sha256 != expected_qc_implementation:
                raise StageIntegrityError(
                    f"state-point QC implementation mismatch: {analysis_state_id}"
                )
            expected_policy_identities = [
                {
                    "role": str(requirement["role"]),
                    "policy_id": str(requirement["policy_id"]),
                    "policy_sha256": str(requirement["policy_sha256"]),
                }
                for requirement in requirements
            ]
            actual_policy_identities = [
                {
                    "role": item["role"],
                    "policy_id": item["policy_id"],
                    "policy_sha256": item["policy_sha256"],
                }
                for item in qc_policy_results
            ]
            if actual_policy_identities != expected_policy_identities:
                raise StageIntegrityError(
                    f"state-point QC policy mismatch: {analysis_state_id}"
                )
            qc_artifact = artifact_record(qc_path, relative_to=incomplete)
            qc_artifact.update(
                {"phase": "quality_control", "role": "state_point_qc"}
            )
            state_record = {
                "state_point_id": analysis_state_id,
                "physical_state_id": physical_state_id,
                "replica_id": replica_id,
                "status": result.execution_status,
                "validated": result.execution_status == "COMPLETE",
                "qc_status": result.qc_status,
                "qc_implementation_sha256": qc_implementation_sha256,
                "qc_policy_results": qc_policy_results,
                "stage_spec_sha256": canonical_sha256(
                    _read_json(stage_dir / "stage_spec.json")
                ),
                "roles": list(item["roles"]),
                "use_for_tg_fit": bool(state["use_for_tg_fit"]),
                "use_for_density": bool(state["use_for_density"]),
                "branch": stage_spec["branch"],
                "ensemble": "npt",
                "target_temperature_K": state["temperature_start_k"],
                "target_pressure_bar": state["pressure_start_bar"],
                "density_context": (
                    resolved["density_context"]
                    if state.get("use_for_density") is True
                    else None
                ),
                "input_snapshot_sha256": replica_inputs[replica_id]["identity"][
                    "snapshot_sha256"
                ],
                "snapshot_class": replica_inputs[replica_id]["identity"][
                    "snapshot_class"
                ],
                "packing_id": replica_inputs[replica_id]["identity"]["packing_id"],
                "expected_atom_count": replica_inputs[replica_id]["identity"][
                    "atom_count"
                ],
                "history_key": stage_spec["history_key"],
                "trajectory_key": trajectory_key,
                "density_reuse_key": density_reuse_key,
                "density_qc_implementation_sha256": None,
                "state_point_provenance": None,
                "run_class": resolved["run_class"],
                "scientific_eligible": resolved["scientific_eligible"],
                "reuse_eligible": resolved["reuse_eligible"],
                "temperature_ramp_start_K": stage_spec[
                    "temperature_ramp_start_k"
                ],
                "temperature_ramp_end_K": stage_spec["temperature_ramp_end_k"],
                "ramp_steps": state_ramp_steps,
                "constant_equilibration_steps": state_equil_steps,
                "production_steps": state_prod_steps,
                "phase_durations_ps": {
                    "ramp_ps": float(state_ramp_steps) * float(engine["dt_ps"]),
                    "equilibration_ps": float(state_equil_steps) * float(engine["dt_ps"]),
                    "production_ps": float(state_prod_steps) * float(engine["dt_ps"]),
                } if resolved.get("sampling_continuation") is not None else dict(item["phase_durations_ps"]),
                "artifacts": {
                    "thermo_samples": production_artifact,
                    "state_point_qc": qc_artifact,
                },
            }
            if resolved.get("sampling_continuation") is not None:
                state_record["analyzed_production_ps"] = sum(
                    _read_json(verify_artifact_record(record, relative_to=incomplete))["duration_ps"]
                    for record in sampling_histories[result.stage_id]
                )
                state_record["sampling_analysis"] = "CUMULATIVE_POST_EQUILIBRATION_PRODUCTION"
            if density_reuse_key is not None:
                density_qc_results = [
                    item
                    for item in qc_policy_results
                    if item["role"] == "density_target"
                ]
                if len(density_qc_results) != 1:
                    raise StageIntegrityError(
                        f"density target lacks one density QC result: {analysis_state_id}"
                    )
                density_qc_result = density_qc_results[0]
                density_qc_code_sha256 = qc_implementation_sha256
                state_record["density_qc_implementation_sha256"] = (
                    density_qc_code_sha256
                )
                qc_status_enum = {
                    "PASS": QCStatus.PASSED,
                    "FAIL": QCStatus.FAILED,
                }[density_qc_result["status"]]
                state_provenance = StatePointProvenance(
                    run_id=identifier,
                    replica_id=replica_id,
                    state_point_id=analysis_state_id,
                    trajectory_key=trajectory_key,
                    density_reuse_key=density_reuse_key,
                    density_qc_policy_id=density_qc_result["policy_id"],
                    density_qc_policy_sha256=density_qc_result[
                        "policy_sha256"
                    ],
                    density_qc_implementation_sha256=(
                        density_qc_code_sha256
                    ),
                    history_key=stage_spec["history_key"],
                    run_manifest_relpath=os.path.relpath(
                        incomplete / "manifest" / "run_manifest.json",
                        start=stage_dir,
                    ),
                    input_snapshot_sha256=replica_inputs[replica_id]["snapshot"][
                        "source_sha256"
                    ],
                    model_sha256=model_identity["sha256"],
                    run_class=str(resolved["run_class"]),
                    scientific_eligible=bool(resolved["scientific_eligible"]),
                    reuse_eligible=bool(resolved["reuse_eligible"]),
                    execution_status=ExecutionStatus.SUCCEEDED,
                    qc_status=qc_status_enum,
                    analysis_status=AnalysisStatus.NOT_APPLICABLE,
                    artifacts=(
                        ArtifactProvenance.from_file(
                            "production_samples",
                            production,
                            relative_to=stage_dir,
                        ),
                        ArtifactProvenance.from_file(
                            "density_qc_result",
                            qc_path,
                            relative_to=stage_dir,
                        ),
                    ),
                )
                state_provenance_path = stage_dir / "state_point_provenance.json"
                write_state_point_provenance(
                    state_provenance_path, state_provenance
                )
                state_record["state_point_provenance"] = artifact_record(
                    state_provenance_path, relative_to=incomplete
                )
            state_records.append(state_record)
            predecessor = stage_dir / "restart.final"
            next_stage_start_step = int(stage_spec["start_step"]) + state_preproduction_steps + state_prod_steps
            history_by_replica[replica_id].append(sha256_file(predecessor))

    # Pin the final QC/finalization decision to the same model, executable,
    # orchestration code, convergence code, and policy files used at launch.
    _verify_execution_identity(resolved, input_root)
    branch = "cooling" if resolved["thermal_mode"] == "tg" else "hold"
    run_manifest = {
        "run_id": identifier,
        "thermal_mode": resolved["thermal_mode"],
        "run_class": resolved["run_class"],
        "scientific_eligible": resolved["scientific_eligible"],
        "reuse_eligible": resolved["reuse_eligible"],
        "run_dir": ".",
        "system": {
            "system_key": canonical_sha256(
                {
                    "replica_snapshot_sha256": {
                        replica_id: item["snapshot"]["source_sha256"]
                        for replica_id, item in sorted(replica_inputs.items())
                    },
                    "polymer_id": system.get("polymer_id"),
                }
            ),
            "polymer_id": system.get("polymer_id"),
            "atom_type_to_element": list(system.get("mace_elements", [])),
            "element_list": list(system.get("element_list", [])),
            "input_snapshots": {
                replica_id: {
                    "source_path": item["snapshot"]["source_path"],
                    "source_sha256": item["snapshot"]["source_sha256"],
                    "snapshot_class": item["identity"]["snapshot_class"],
                    "packing_id": item["identity"]["packing_id"],
                    "atom_count": item["identity"]["atom_count"],
                    "atom_type_count": item["identity"]["atom_type_count"],
                    "element_counts": item["identity"]["element_counts"],
                    "snapshot_artifact": artifact_record(
                        Path(item["copied_snapshot_path"]), relative_to=incomplete
                    ),
                    "metadata_artifact": artifact_record(
                        Path(item["metadata"]["copied_path"]), relative_to=incomplete
                    ),
                }
                for replica_id, item in sorted(replica_inputs.items())
            },
        },
        "potential": model_identity,
        "engine": resolved["execution_identity"]["engine"],
        "execution_identity": resolved["execution_identity"],
        "replicas": [
            {
                "replica_id": str(replica["replica_id"]),
                "seed": int(replica["seed"]),
            }
            for replica in resolved["replicas"]
        ],
        "protocol": {
            "ensemble": "npt",
            "branch": branch,
            "density_context": resolved["density_context"],
            "protocol_sha256": resolved["protocol_sha256"],
        },
        "resolved_run_spec": artifact_record(spec_path, relative_to=incomplete),
        "runtime_inputs": [
            {
                "source_path": item["source_path"],
                "source_sha256": item["source_sha256"],
                "artifact": artifact_record(
                    Path(item["copied_path"]), relative_to=incomplete
                ),
            }
            for item in snapshotted_inputs
        ],
        "state_points": state_records,
        "initialization_stages": initialization_records,
    }
    if parent_identity is not None:
        run_manifest["density_parent_restart"] = {
            "identity": parent_identity,
            "copied_artifacts": parent_input_records,
            "interpretation": "UNQUALIFIED_PARENT_NEW_TRANSITION_QC_REQUIRED",
        }
    if resolved.get("sampling_continuation") is not None:
        run_manifest["sampling_continuation"] = resolved["sampling_continuation"]
        run_manifest["sampling_history"] = sampling_histories
    if output_ownership is not None:
        run_manifest["submission_claim"] = {
            "claim_uuid": output_ownership.claim_uuid,
            "submission_attempt_number": output_ownership.attempt_number,
            "submission_attempt_uuid": output_ownership.attempt_uuid,
            "output_owner_uuid": output_ownership.owner_uuid,
            "execution_resolved_spec_sha256": (
                output_ownership.execution_resolved_spec_sha256
            ),
            "resume": output_ownership.resume,
        }
    return finalize_run_directory(
        incomplete_dir=incomplete,
        final_dir=final,
        run_manifest=run_manifest,
        stage_results=state_results,
        initialization_results=initialization_results,
    )
