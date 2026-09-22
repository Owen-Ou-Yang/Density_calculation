"""Race-free scheduler submission and output ownership for thermal runs.

The claim directory is the shared-filesystem lock.  It is acquired with one
``os.mkdir`` and is never deleted.  Mutable state is atomically replaced after
an immutable, hash-chained history record has been durably installed.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import errno
import fcntl
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import stat
import subprocess
import time
from typing import Any, Callable, Iterator, Mapping, Sequence
import uuid

from .provenance import canonical_sha256, sha256_file


CLAIM_SCHEMA = "thermal-properties-submission-claim/v1"
STATE_SCHEMA = "thermal-properties-submission-claim-state/v1"
HISTORY_SCHEMA = "thermal-properties-submission-claim-history/v1"
OWNER_SCHEMA = "thermal-properties-output-owner/v1"
CLAIM_ROOT_NAME = ".submission_claims"
OWNER_MARKER_NAME = ".thermal_output_owner.json"
OWNER_LEASE_NAME = ".output_owner.lock"
SCHEDULER_EVIDENCE_MAX_AGE_SECONDS = 300.0
SCHEDULER_EVIDENCE_MAX_FUTURE_SKEW_SECONDS = 30.0
_UUID_NAMESPACE = uuid.UUID("671fc3e0-a1d4-48f2-bfef-367196301123")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REASON_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_EVENT_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SGE_JOB_RE = re.compile(
    r'\AYour job[ \t]+(\d+)(?:[ \t]+\("[^"\n]*"\))?'
    r'[ \t]+has been submitted[ \t]*\n?\Z'
)
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class SubmissionClaimError(RuntimeError):
    """Base class for fail-closed claim errors."""


class ClaimValidationError(SubmissionClaimError):
    """A claim identity, path, or artifact is unsafe or malformed."""


class ClaimBlockedError(SubmissionClaimError):
    """A valid existing claim or active owner blocks the operation."""


class ClaimConflictError(ClaimBlockedError):
    """The same run ID is already bound to a different specification."""


class ClaimTransitionError(SubmissionClaimError):
    """The requested state transition is not allowed."""


class ClaimState(str, Enum):
    RESERVED = "RESERVED"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    SUBMISSION_FAILED = "SUBMISSION_FAILED"
    SUBMISSION_UNKNOWN = "SUBMISSION_UNKNOWN"
    INTERRUPTED = "INTERRUPTED"
    FAILED = "FAILED"
    BLOCKED_CONFLICT = "BLOCKED_CONFLICT"
    RELEASED = "RELEASED"
    RECONCILED = "RECONCILED"


class SchedulerEvidence(str, Enum):
    ACTIVE = "ACTIVE"
    ABSENT = "ABSENT"
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"


_TRANSITIONS: dict[ClaimState, frozenset[ClaimState]] = {
    ClaimState.RESERVED: frozenset(
        {
            ClaimState.SUBMITTED,
            ClaimState.SUBMISSION_FAILED,
            ClaimState.SUBMISSION_UNKNOWN,
            ClaimState.BLOCKED_CONFLICT,
            ClaimState.RECONCILED,
            ClaimState.RELEASED,
        }
    ),
    ClaimState.SUBMITTED: frozenset(
        {
            ClaimState.RUNNING,
            ClaimState.INTERRUPTED,
            ClaimState.SUBMISSION_UNKNOWN,
            ClaimState.BLOCKED_CONFLICT,
            ClaimState.RECONCILED,
            ClaimState.RELEASED,
        }
    ),
    ClaimState.RUNNING: frozenset(
        {
            ClaimState.COMPLETE,
            ClaimState.INTERRUPTED,
            ClaimState.FAILED,
            ClaimState.BLOCKED_CONFLICT,
            ClaimState.RECONCILED,
        }
    ),
    ClaimState.SUBMISSION_FAILED: frozenset(
        {
            ClaimState.RESERVED,
            ClaimState.SUBMISSION_UNKNOWN,
            ClaimState.RELEASED,
            ClaimState.RECONCILED,
        }
    ),
    ClaimState.SUBMISSION_UNKNOWN: frozenset(
        {ClaimState.RECONCILED, ClaimState.RELEASED, ClaimState.BLOCKED_CONFLICT}
    ),
    ClaimState.INTERRUPTED: frozenset(
        {ClaimState.RESERVED, ClaimState.RECONCILED, ClaimState.RELEASED}
    ),
    ClaimState.FAILED: frozenset(
        {ClaimState.RESERVED, ClaimState.RECONCILED, ClaimState.RELEASED}
    ),
    ClaimState.BLOCKED_CONFLICT: frozenset(
        {ClaimState.RECONCILED, ClaimState.RELEASED}
    ),
    ClaimState.RECONCILED: frozenset(
        {ClaimState.RESERVED, ClaimState.RELEASED}
    ),
    ClaimState.COMPLETE: frozenset(),
    ClaimState.RELEASED: frozenset(),
}


IMPLEMENTATION_RELATIVE_FILES = (
    "__init__.py",
    "cli.py",
    "quick_job.py",
    "config.py",
    "planning.py",
    "provenance.py",
    "reuse.py",
    "simulation.py",
    "snapshot_contract.py",
    "submission_claim.py",
    "run_thermal_snapshot.sh",
    "thermal_job.sh",
    "scheduler/submit_thermal_snapshot.sh",
    "scheduler/submit_thermal_batch.sh",
    "analysis/__init__.py",
    "analysis/convergence.py",
    "analysis/samples.py",
    "lammps/in.initialize_mace_mh1.lmp",
    "lammps/in.npt_stage_mace_mh1.lmp",
    "lammps/mace_mh1_thermal_setup.mod",
    "lammps/thermal_temperature.mod",
    "config/thermal_schema_v1.json",
    "config/density_qc_v1.json",
    "config/mace_transition_qc_v1.json",
    "config/tg_fit_v1.json",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_sha256(value: object, field: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.lower()):
        raise ClaimValidationError(f"{field} must be a 64-character SHA-256")
    return value.lower()


def validate_run_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or value in {"", ".", ".."}
        or not _RUN_ID_RE.fullmatch(value)
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ClaimValidationError(
            "run_id must match [A-Za-z0-9][A-Za-z0-9_.-]* and cannot be . or .."
        )
    return value


def _validate_reason(value: object) -> str:
    if not isinstance(value, str) or not _REASON_RE.fullmatch(value):
        raise ClaimValidationError(
            "reason must be a nonempty machine-readable identifier"
        )
    return value


def _validate_uuid(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ClaimValidationError(f"{field} must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ClaimValidationError(f"{field} must be a canonical UUID") from exc
    canonical = str(parsed)
    if value != canonical:
        raise ClaimValidationError(f"{field} must be a canonical UUID")
    return canonical


def _validate_scheduler_job_name(value: object, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 255
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ClaimValidationError("scheduler job name is unsafe or malformed")
    return value


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _reject_symlink(path: Path, field: str) -> None:
    if _lexists(path) and stat.S_ISLNK(os.lstat(path).st_mode):
        raise ClaimValidationError(f"{field} cannot be a symlink: {path}")


def canonical_run_root(value: str | Path, *, must_exist: bool) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise ClaimValidationError(
            "thermal run root must be absolute; claim identity cannot depend on cwd"
        )
    _reject_symlink(raw, "thermal run root")
    root = raw.resolve(strict=False)
    if must_exist:
        if not root.is_dir():
            raise ClaimValidationError(f"thermal run root is not a directory: {root}")
        _reject_symlink(root, "canonical thermal run root")
    return root


def _contained(path: Path, root: Path, field: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ClaimValidationError(f"{field} escapes canonical run root: {path}") from exc
    return path


def build_thermal_implementation_inventory(
    package_root: str | Path | None = None,
) -> dict[str, Any]:
    root = (
        Path(package_root).resolve()
        if package_root is not None
        else Path(__file__).resolve().parent
    )
    files: dict[str, dict[str, Any]] = {}
    for relative_name in IMPLEMENTATION_RELATIVE_FILES:
        source = root / relative_name
        _reject_symlink(source, f"thermal implementation file {relative_name}")
        if not source.is_file():
            raise ClaimValidationError(
                f"thermal implementation file is missing: {source}"
            )
        files[relative_name] = {
            "sha256": sha256_file(source),
            "bytes": source.stat().st_size,
        }
    return {
        "schema_version": "thermal-properties-implementation-inventory/v1",
        "files": files,
        "sha256": canonical_sha256(files),
    }


def _claim_uuid_for(run_root: Path, run_id: str, base_spec_hash: str) -> str:
    identity_text = json.dumps(
        {
            "run_root": str(run_root),
            "run_id": run_id,
            "base_resolved_spec_sha256": base_spec_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return str(uuid.uuid5(_UUID_NAMESPACE, identity_text))


@dataclass(frozen=True)
class ClaimPlan:
    run_root: Path
    run_id: str
    claim_root: Path
    claim_dir: Path
    expected_output_path: Path
    incomplete_output_path: Path
    claim_uuid: str
    base_resolved_spec_sha256: str
    execution_resolved_spec_sha256: str | None
    frozen_config_sha256: str | None
    thermal_implementation_inventory_sha256: str | None
    job_script_sha256: str | None
    run_class: str
    scientific_eligible: bool
    reuse_eligible: bool
    scheduler_job_name: str | None

    def __post_init__(self) -> None:
        _validate_plan_geometry(self)

    @property
    def complete_for_acquisition(self) -> bool:
        return all(
            value is not None
            for value in (
                self.execution_resolved_spec_sha256,
                self.frozen_config_sha256,
                self.thermal_implementation_inventory_sha256,
                self.job_script_sha256,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_root": str(self.run_root),
            "run_id": self.run_id,
            "claim_root": str(self.claim_root),
            "claim_dir": str(self.claim_dir),
            "expected_output_path": str(self.expected_output_path),
            "incomplete_output_path": str(self.incomplete_output_path),
            "claim_uuid": self.claim_uuid,
            "base_resolved_spec_sha256": self.base_resolved_spec_sha256,
            "execution_resolved_spec_sha256": self.execution_resolved_spec_sha256,
            "frozen_config_sha256": self.frozen_config_sha256,
            "thermal_implementation_inventory_sha256": (
                self.thermal_implementation_inventory_sha256
            ),
            "job_script_sha256": self.job_script_sha256,
            "run_class": self.run_class,
            "scientific_eligible": self.scientific_eligible,
            "reuse_eligible": self.reuse_eligible,
            "scheduler_job_name": self.scheduler_job_name,
        }


def _validate_plan_geometry(plan: ClaimPlan) -> None:
    """Re-derive every claim/output path and identity from its canonical inputs."""

    root = canonical_run_root(plan.run_root, must_exist=False)
    identifier = validate_run_id(plan.run_id)
    if not isinstance(plan.run_root, Path) or plan.run_root != root:
        raise ClaimValidationError("claim plan run root is not canonical")
    expected_paths = {
        "claim_root": root / CLAIM_ROOT_NAME,
        "claim_dir": root / CLAIM_ROOT_NAME / f"{identifier}.claim",
        "expected_output_path": root / identifier,
        "incomplete_output_path": root / f"{identifier}.incomplete",
    }
    for field_name, expected_path in expected_paths.items():
        actual_path = getattr(plan, field_name)
        if not isinstance(actual_path, Path) or actual_path != expected_path:
            raise ClaimValidationError(
                f"claim plan {field_name} is not its exact canonical derived path"
            )
        _contained(actual_path, root, field_name.replace("_", " "))
    base_hash = _validate_sha256(
        plan.base_resolved_spec_sha256, "base_resolved_spec_sha256"
    )
    _validate_sha256(
        plan.execution_resolved_spec_sha256,
        "execution_resolved_spec_sha256",
        optional=True,
    )
    _validate_sha256(
        plan.frozen_config_sha256, "frozen_config_sha256", optional=True
    )
    _validate_sha256(
        plan.thermal_implementation_inventory_sha256,
        "thermal_implementation_inventory_sha256",
        optional=True,
    )
    _validate_sha256(plan.job_script_sha256, "job_script_sha256", optional=True)
    if not isinstance(plan.run_class, str) or not plan.run_class:
        raise ClaimValidationError("run_class must be a nonempty string")
    if type(plan.scientific_eligible) is not bool or type(plan.reuse_eligible) is not bool:
        raise ClaimValidationError("eligibility flags must be explicit booleans")
    _validate_scheduler_job_name(plan.scheduler_job_name, optional=True)
    exact_claim_uuid = _validate_uuid(plan.claim_uuid, "claim UUID")
    if exact_claim_uuid != _claim_uuid_for(root, identifier, base_hash or ""):
        raise ClaimValidationError("claim UUID is not derived from the canonical claim identity")


def plan_claim(
    *,
    run_root: str | Path,
    run_id: str,
    base_resolved_spec_sha256: str,
    execution_resolved_spec_sha256: str | None,
    frozen_config_sha256: str | None,
    thermal_implementation_inventory_sha256: str | None,
    job_script_sha256: str | None,
    run_class: str,
    scientific_eligible: bool,
    reuse_eligible: bool,
    scheduler_job_name: str | None = None,
) -> ClaimPlan:
    root = canonical_run_root(run_root, must_exist=False)
    identifier = validate_run_id(run_id)
    base_hash = _validate_sha256(
        base_resolved_spec_sha256, "base_resolved_spec_sha256"
    )
    execution_hash = _validate_sha256(
        execution_resolved_spec_sha256,
        "execution_resolved_spec_sha256",
        optional=True,
    )
    config_hash = _validate_sha256(
        frozen_config_sha256, "frozen_config_sha256", optional=True
    )
    implementation_hash = _validate_sha256(
        thermal_implementation_inventory_sha256,
        "thermal_implementation_inventory_sha256",
        optional=True,
    )
    script_hash = _validate_sha256(
        job_script_sha256, "job_script_sha256", optional=True
    )
    if not isinstance(run_class, str) or not run_class:
        raise ClaimValidationError("run_class must be a nonempty string")
    if type(scientific_eligible) is not bool or type(reuse_eligible) is not bool:
        raise ClaimValidationError("eligibility flags must be explicit booleans")
    job_name = _validate_scheduler_job_name(scheduler_job_name, optional=True)
    claim_root = _contained(root / CLAIM_ROOT_NAME, root, "claim root")
    claim_dir = _contained(
        claim_root / f"{identifier}.claim", root, "claim directory"
    )
    expected = _contained(root / identifier, root, "expected output")
    incomplete = _contained(
        root / f"{identifier}.incomplete", root, "incomplete output"
    )
    claim_uuid = _claim_uuid_for(root, identifier, base_hash or "")
    return ClaimPlan(
        run_root=root,
        run_id=identifier,
        claim_root=claim_root,
        claim_dir=claim_dir,
        expected_output_path=expected,
        incomplete_output_path=incomplete,
        claim_uuid=claim_uuid,
        base_resolved_spec_sha256=base_hash or "",
        execution_resolved_spec_sha256=execution_hash,
        frozen_config_sha256=config_hash,
        thermal_implementation_inventory_sha256=implementation_hash,
        job_script_sha256=script_hash,
        run_class=run_class,
        scientific_eligible=scientific_eligible,
        reuse_eligible=reuse_eligible,
        scheduler_job_name=job_name,
    )


def _attempt_uuid(claim_uuid: str, number: int) -> str:
    return str(uuid.uuid5(uuid.UUID(claim_uuid), f"submission-attempt:{number}"))


def planned_submission_attempt(plan: ClaimPlan, number: int = 1) -> dict[str, Any]:
    _validate_plan_geometry(plan)
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        raise ClaimValidationError("submission attempt number must be positive")
    return {
        "attempt_number": number,
        "attempt_uuid": _attempt_uuid(plan.claim_uuid, number),
    }


@dataclass(frozen=True)
class ClaimHandle:
    plan: ClaimPlan
    attempt_number: int
    attempt_uuid: str
    submission_kind: str
    reservation_token: str = field(repr=False)


@dataclass(frozen=True)
class SchedulerObservation:
    status: SchedulerEvidence
    observed_at_utc: str
    command_available: bool
    returncode: int | None
    stdout: str
    stderr: str
    matching_job_ids: tuple[str, ...]
    reason: str
    expected_job_id: str | None = None
    expected_job_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "observed_at_utc": self.observed_at_utc,
            "command_available": self.command_available,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "matching_job_ids": list(self.matching_job_ids),
            "reason": self.reason,
            "expected_job_id": self.expected_job_id,
            "expected_job_name": self.expected_job_name,
        }


@dataclass
class _OutputOwnerLease:
    path: Path
    descriptor: int
    released: bool = False


@dataclass(frozen=True)
class OutputOwnership:
    claim_dir: Path
    claim_uuid: str
    run_root: Path
    run_id: str
    expected_output_path: Path
    incomplete_output_path: Path
    execution_resolved_spec_sha256: str
    attempt_number: int
    attempt_uuid: str
    owner_uuid: str
    resume: bool
    owner_lease: _OutputOwnerLease = field(repr=False, compare=False)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _encoded_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _reject_symlink(path.parent, "JSON parent")
    if not path.parent.is_dir():
        raise ClaimValidationError(f"JSON parent is missing: {path.parent}")
    _reject_symlink(path, "JSON destination")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(_encoded_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if _lexists(temporary):
            os.unlink(temporary)


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    _reject_symlink(path.parent, "immutable JSON parent")
    if not path.parent.is_dir():
        raise ClaimValidationError(
            f"immutable JSON parent is missing: {path.parent}"
        )
    if _lexists(path):
        raise ClaimValidationError(f"immutable JSON already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(_encoded_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o400)
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if _lexists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> dict[str, Any]:
    _reject_symlink(path, "claim JSON")
    if not path.is_file():
        raise ClaimValidationError(f"claim JSON is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClaimValidationError(f"cannot read claim JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ClaimValidationError(f"claim JSON root is not an object: {path}")
    return payload


def _mkdir_exact(path: Path, *, mode: int = 0o700) -> None:
    os.mkdir(path, mode)
    _fsync_directory(path.parent)


def _validate_claim_container(claim_dir: Path) -> None:
    claim_root = claim_dir.parent
    _reject_symlink(claim_root, "claim root")
    if not claim_root.is_dir():
        raise ClaimValidationError(f"claim root is missing: {claim_root}")
    _reject_symlink(claim_dir, "claim directory")
    if not claim_dir.is_dir():
        raise ClaimValidationError(f"claim directory is missing: {claim_dir}")


@contextmanager
def _transition_lock(claim_dir: Path, *, wait_seconds: float = 0.0) -> Iterator[None]:
    _validate_claim_container(claim_dir)
    if (
        isinstance(wait_seconds, bool)
        or not isinstance(wait_seconds, (int, float))
        or not math.isfinite(float(wait_seconds))
        or wait_seconds < 0
    ):
        raise ClaimValidationError("transition-lock wait must be finite and nonnegative")
    lock = claim_dir / ".transition.lock"
    _reject_symlink(lock, "transition lock")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock, flags, 0o600)
    except OSError as exc:
        raise ClaimValidationError(f"cannot open transition lock: {lock}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ClaimValidationError("transition lock is not a regular file")
        deadline = time.monotonic() + float(wait_seconds)
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise ClaimValidationError("cannot acquire transition lock") from exc
                if time.monotonic() >= deadline:
                    raise ClaimBlockedError(
                        f"claim transition is already active: {lock}"
                    ) from exc
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _acquire_output_owner_lease(
    claim_dir: Path, *, wait_seconds: float = 0.0
) -> _OutputOwnerLease:
    """Acquire the process-lifetime lease that proves an output owner is dead."""

    _validate_claim_container(claim_dir)
    if (
        isinstance(wait_seconds, bool)
        or not isinstance(wait_seconds, (int, float))
        or not math.isfinite(float(wait_seconds))
        or wait_seconds < 0
    ):
        raise ClaimValidationError("owner-lease wait must be finite and nonnegative")
    path = claim_dir / OWNER_LEASE_NAME
    _reject_symlink(path, "output owner lease")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ClaimValidationError(f"cannot open output owner lease: {path}") from exc
    acquired = False
    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = os.lstat(path)
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or not stat.S_ISREG(path_stat.st_mode)
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise ClaimValidationError("output owner lease is not the exact regular file")
        os.fchmod(descriptor, 0o600)
        deadline = time.monotonic() + float(wait_seconds)
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise ClaimValidationError(
                        "cannot acquire output owner lease"
                    ) from exc
                if time.monotonic() >= deadline:
                    raise ClaimBlockedError(
                        "a live process still holds the output owner lease"
                    ) from exc
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        os.fsync(descriptor)
        _fsync_directory(claim_dir)
        return _OutputOwnerLease(path=path, descriptor=descriptor)
    except BaseException:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        raise


def _verify_output_owner_lease(ownership: OutputOwnership) -> None:
    lease = ownership.owner_lease
    if lease.released:
        raise ClaimBlockedError("output owner lease was already released")
    try:
        descriptor_stat = os.fstat(lease.descriptor)
        path_stat = os.lstat(lease.path)
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or not stat.S_ISREG(path_stat.st_mode)
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise ClaimValidationError("output owner lease identity changed")
        fcntl.flock(lease.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise ClaimBlockedError("output owner lease is no longer held") from exc


def _release_owner_lease(lease: _OutputOwnerLease) -> None:
    if lease.released:
        return
    try:
        fcntl.flock(lease.descriptor, fcntl.LOCK_UN)
    finally:
        try:
            os.close(lease.descriptor)
        finally:
            lease.released = True


def _release_output_owner_lease(ownership: OutputOwnership) -> None:
    """Private process-exit helper; it does not alter persisted claim state."""

    _release_owner_lease(ownership.owner_lease)


def _load_claim(claim_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_claim_container(claim_dir)
    identity = _read_json(claim_dir / "identity.json")
    state = _read_json(claim_dir / "state.json")
    if identity.get("schema_version") != CLAIM_SCHEMA:
        raise ClaimValidationError("unsupported claim identity schema")
    if state.get("schema_version") != STATE_SCHEMA:
        raise ClaimValidationError("unsupported claim state schema")
    if identity.get("claim_uuid") != state.get("claim_uuid"):
        raise ClaimValidationError("claim identity/state UUID mismatch")
    claim_uuid = _validate_uuid(identity.get("claim_uuid"), "claim UUID")
    try:
        ClaimState(str(state.get("state")))
    except ValueError as exc:
        raise ClaimValidationError("claim state is unknown") from exc
    head_name = state.get("history_head")
    head_hash = state.get("history_head_sha256")
    if not isinstance(head_name, str) or not isinstance(head_hash, str):
        raise ClaimValidationError("claim state lacks immutable history head")
    history = claim_dir / "history"
    _reject_symlink(history, "claim history directory")
    if not history.is_dir():
        raise ClaimValidationError("claim history directory is missing")
    try:
        sequence = int(state.get("history_sequence"))
    except (TypeError, ValueError) as exc:
        raise ClaimValidationError("claim history sequence is malformed") from exc
    if sequence <= 0 or Path(head_name).name != head_name:
        raise ClaimValidationError("claim history head is malformed")
    reverse_chain: list[tuple[Path, dict[str, Any]]] = []
    current_name: str | None = head_name
    current_hash: str | None = head_hash
    for expected_sequence in range(sequence, 0, -1):
        if current_name is None or Path(current_name).name != current_name:
            raise ClaimValidationError("claim history predecessor name is malformed")
        record_path = history / current_name
        _reject_symlink(record_path, "claim history record")
        record = _read_json(record_path)
        if sha256_file(record_path) != current_hash:
            raise ClaimValidationError("claim history record hash mismatch")
        if record.get("schema_version") != HISTORY_SCHEMA:
            raise ClaimValidationError("unsupported claim history schema")
        if record.get("sequence") != expected_sequence:
            raise ClaimValidationError("claim history sequence mismatch")
        if record.get("claim_uuid") != claim_uuid:
            raise ClaimValidationError("claim history UUID mismatch")
        reverse_chain.append((record_path, record))
        current_name = record.get("previous_history_name")
        current_hash = record.get("previous_history_sha256")
    if current_name is not None or current_hash is not None:
        raise ClaimValidationError("claim history chain does not terminate")

    previous_hash: str | None = None
    previous_name: str | None = None
    previous_state: ClaimState | None = None
    forward_chain = list(reversed(reverse_chain))
    for number, (record_path, record) in enumerate(forward_chain, start=1):
        if (
            record.get("previous_history_name") != previous_name
            or record.get("previous_history_sha256") != previous_hash
        ):
            raise ClaimValidationError("claim history hash chain mismatch")
        from_raw = record.get("from_state")
        to_raw = record.get("to_state")
        try:
            to_state = ClaimState(str(to_raw))
            from_state = None if from_raw is None else ClaimState(str(from_raw))
        except ValueError as exc:
            raise ClaimValidationError("claim history state is unknown") from exc
        if number == 1:
            if from_state is not None or to_state is not ClaimState.RESERVED:
                raise ClaimValidationError("claim history must begin with RESERVED")
            details = record.get("details")
            if not isinstance(details, dict) or details.get(
                "identity_sha256"
            ) != sha256_file(claim_dir / "identity.json"):
                raise ClaimValidationError("claim history does not bind identity")
        else:
            if from_state is not previous_state:
                raise ClaimValidationError("claim history state chain mismatch")
            event_type = record.get("event_type")
            if to_state is from_state:
                allowed_same_state_event = (
                    from_state is ClaimState.RESERVED
                    and event_type == "QSUB_INVOCATION_RECORDED"
                ) or (
                    from_state is ClaimState.SUBMITTED
                    and event_type == "OUTPUT_OWNER_ACQUISITION_STARTED"
                ) or (
                    from_state in {ClaimState.INTERRUPTED, ClaimState.FAILED}
                    and event_type
                    in {
                        "ORPHANED_OUTPUT_OWNER_REVOCATION_STARTED",
                        "ORPHANED_OUTPUT_OWNER_REVOKED",
                    }
                )
                if not allowed_same_state_event:
                    raise ClaimValidationError("unauthorized same-state history event")
            elif from_state is None or to_state not in _TRANSITIONS[from_state]:
                raise ClaimValidationError("claim history contains an invalid transition")
        previous_hash = sha256_file(record_path)
        previous_name = record_path.name
        previous_state = to_state
    if previous_name != head_name or previous_hash != head_hash:
        raise ClaimValidationError("claim history head hash mismatch")
    if previous_state is not ClaimState(str(state["state"])):
        raise ClaimValidationError("claim state disagrees with history head")
    snapshot = dict(state)
    snapshot.pop("history_head_sha256", None)
    last_record = forward_chain[-1][1] if forward_chain else None
    if last_record is None or last_record.get(
        "state_snapshot_sha256"
    ) != canonical_sha256(snapshot):
        raise ClaimValidationError("claim state snapshot hash mismatch")
    return identity, state


def _commit_state(
    claim_dir: Path,
    state: dict[str, Any],
    *,
    previous_state: ClaimState | None,
    event_type: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(event_type, str) or not _EVENT_RE.fullmatch(event_type):
        raise ClaimValidationError("claim event_type is unsafe or malformed")
    sequence = int(state.get("history_sequence", 0)) + 1
    current = ClaimState(str(state["state"]))
    timestamp = _utc_now()
    history_name = (
        f"{sequence:06d}_{event_type.lower()}_{uuid.uuid4().hex}.json"
    )
    committed_state = dict(state)
    committed_state.update(
        {
            "updated_at_utc": timestamp,
            "history_sequence": sequence,
            "history_head": history_name,
        }
    )
    committed_state.pop("history_head_sha256", None)
    history = {
        "schema_version": HISTORY_SCHEMA,
        "sequence": sequence,
        "claim_uuid": state["claim_uuid"],
        "event_type": event_type,
        "from_state": previous_state.value if previous_state is not None else None,
        "to_state": current.value,
        "occurred_at_utc": timestamp,
        "previous_history_name": state.get("history_head"),
        "previous_history_sha256": state.get("history_head_sha256"),
        "state_snapshot_sha256": canonical_sha256(committed_state),
        "details": dict(details or {}),
    }
    history_path = claim_dir / "history" / history_name
    _write_immutable_json(history_path, history)
    committed_state["history_head_sha256"] = sha256_file(history_path)
    state.clear()
    state.update(committed_state)
    _atomic_write_json(claim_dir / "state.json", state)
    return state


def _validated_submission_receipt(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the current attempt only when it records a real qsub receipt."""

    attempt_uuid = state.get("current_attempt_uuid")
    attempts = state.get("attempts")
    if not isinstance(attempt_uuid, str) or not isinstance(attempts, list):
        raise ClaimValidationError("submitted claim has malformed attempt metadata")
    if any(not isinstance(item, dict) for item in attempts):
        raise ClaimValidationError("submitted claim has a malformed attempt entry")
    matches = [item for item in attempts if item.get("attempt_uuid") == attempt_uuid]
    if len(matches) != 1:
        raise ClaimValidationError(
            "submitted claim current attempt is missing or duplicated"
        )
    attempt = matches[0]
    if attempt.get("qsub_invoked") is not True:
        raise ClaimValidationError("submitted claim lacks a recorded qsub invocation")
    if attempt.get("result") != "SUBMITTED":
        raise ClaimValidationError("submitted claim lacks an accepted qsub receipt")
    state_job_id = state.get("scheduler_job_id")
    attempt_job_id = attempt.get("scheduler_job_id")
    if (
        not isinstance(state_job_id, str)
        or not state_job_id.isdigit()
        or attempt_job_id != state_job_id
    ):
        raise ClaimValidationError(
            "submitted claim scheduler job ID disagrees with its qsub receipt"
        )
    return attempt


def _transition_locked(
    claim_dir: Path,
    claim_uuid: str,
    target: ClaimState,
    *,
    event_type: str,
    updates: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
    allow_same: bool = False,
) -> dict[str, Any]:
    identity, state = _load_claim(claim_dir)
    if identity.get("claim_uuid") != claim_uuid:
        raise ClaimValidationError("claim UUID mismatch")
    current = ClaimState(str(state["state"]))
    if target == current:
        if not allow_same:
            raise ClaimTransitionError(f"same-state transition is not allowed: {current.value}")
    elif target not in _TRANSITIONS[current]:
        raise ClaimTransitionError(
            f"transition {current.value} -> {target.value} is not allowed"
        )
    new_state = dict(state)
    if updates:
        protected = {
            "schema_version",
            "claim_uuid",
            "state",
            "created_at_utc",
            "updated_at_utc",
            "history_sequence",
            "history_head",
            "history_head_sha256",
        }
        overlap = protected.intersection(updates)
        if overlap:
            raise ClaimValidationError(
                "claim updates cannot replace protected fields: "
                + ", ".join(sorted(overlap))
            )
        new_state.update(dict(updates))
    new_state["state"] = target.value
    if target is ClaimState.SUBMITTED:
        _validated_submission_receipt(new_state)
    return _commit_state(
        claim_dir,
        new_state,
        previous_state=current,
        event_type=event_type,
        details=details,
    )


def inspect_claim(plan: ClaimPlan) -> dict[str, Any]:
    """Inspect a planned claim without creating or mutating any path."""

    _validate_plan_geometry(plan)
    _reject_symlink(plan.claim_root, "claim root")
    _reject_symlink(plan.claim_dir, "claim directory")
    if not _lexists(plan.claim_dir):
        return {
            "exists": False,
            "blocking": False,
            "claim_path": str(plan.claim_dir),
            "planned_claim_uuid": plan.claim_uuid,
        }
    try:
        identity, state = _load_claim(plan.claim_dir)
        conflict = (
            identity.get("base_resolved_spec_sha256")
            != plan.base_resolved_spec_sha256
            or (
                plan.execution_resolved_spec_sha256 is not None
                and identity.get("execution_resolved_spec_sha256")
                != plan.execution_resolved_spec_sha256
            )
        )
        return {
            "exists": True,
            "blocking": True,
            "conflict": conflict,
            "claim_path": str(plan.claim_dir),
            "claim_uuid": identity.get("claim_uuid"),
            "state": state.get("state"),
            "scheduler_job_id": state.get("scheduler_job_id"),
            "scheduler_job_name": state.get(
                "scheduler_job_name", identity.get("scheduler_job_name")
            ),
            "active_owner_uuid": state.get("active_owner_uuid"),
            "pending_owner_uuid": state.get("pending_owner_uuid"),
            "submission_attempt_number": state.get("submission_attempt_number"),
        }
    except SubmissionClaimError as exc:
        return {
            "exists": True,
            "blocking": True,
            "conflict": True,
            "claim_path": str(plan.claim_dir),
            "integrity_error": str(exc),
        }


def read_scheduler_authorization(
    *,
    run_root: str | Path,
    run_id: str,
    claim_uuid: str,
    attempt_number: int,
    attempt_uuid: str,
    submission_kind: str,
) -> dict[str, Any]:
    """Read the immutable claim authorization named by a scheduler payload."""

    root = canonical_run_root(run_root, must_exist=True)
    identifier = validate_run_id(run_id)
    exact_claim_uuid = _validate_uuid(claim_uuid, "claim UUID")
    exact_attempt_uuid = _validate_uuid(attempt_uuid, "submission attempt UUID")
    if isinstance(attempt_number, bool) or not isinstance(attempt_number, int):
        raise ClaimValidationError("submission attempt number must be an integer")
    if submission_kind not in {"initial", "resume"}:
        raise ClaimValidationError("submission kind is invalid")
    claim_dir = root / CLAIM_ROOT_NAME / f"{identifier}.claim"
    identity, state = _load_claim(claim_dir)
    exact_identity = {
        "claim_uuid": exact_claim_uuid,
        "canonical_run_id": identifier,
        "canonical_thermal_run_root": str(root),
        "canonical_expected_output_path": str(root / identifier),
        "canonical_incomplete_output_path": str(
            root / f"{identifier}.incomplete"
        ),
    }
    for field, value in exact_identity.items():
        if identity.get(field) != value:
            raise ClaimValidationError(
                f"scheduler authorization mismatch: {field}"
            )
    if state.get("current_attempt_uuid") != exact_attempt_uuid or state.get(
        "submission_attempt_number"
    ) != attempt_number:
        raise ClaimValidationError("scheduler authorization attempt mismatch")
    attempts = state.get("attempts")
    if not isinstance(attempts, list) or any(
        not isinstance(item, dict) for item in attempts
    ):
        raise ClaimValidationError("claim attempts are malformed")
    matches = [
        item for item in attempts if item.get("attempt_uuid") == exact_attempt_uuid
    ]
    if len(matches) != 1 or matches[0].get("submission_kind") != submission_kind:
        raise ClaimValidationError("scheduler authorization attempt record mismatch")
    current = ClaimState(str(state.get("state")))
    if current is ClaimState.RESERVED:
        if not _submission_outcome_is_pending(state):
            raise ClaimBlockedError("qsub has not been invoked for this reservation")
    elif current is not ClaimState.SUBMITTED:
        raise ClaimBlockedError(
            f"claim state {current.value} does not authorize scheduler startup"
        )
    return {
        "claim_uuid": exact_claim_uuid,
        "canonical_run_id": identifier,
        "canonical_thermal_run_root": str(root),
        "base_resolved_spec_sha256": identity.get(
            "base_resolved_spec_sha256"
        ),
        "execution_resolved_spec_sha256": identity.get(
            "execution_resolved_spec_sha256"
        ),
        "frozen_config_sha256": identity.get("frozen_config_sha256"),
        "run_class": identity.get("run_class"),
        "scientific_eligible": identity.get("scientific_eligible"),
        "reuse_eligible": identity.get("reuse_eligible"),
        "scheduler_job_id": state.get("scheduler_job_id"),
        "state": current.value,
    }


def _ensure_claim_root(plan: ClaimPlan) -> None:
    _validate_plan_geometry(plan)
    canonical_run_root(plan.run_root, must_exist=True)
    _reject_symlink(plan.claim_root, "claim root")
    if _lexists(plan.claim_root):
        if not plan.claim_root.is_dir():
            raise ClaimValidationError(f"claim root is not a directory: {plan.claim_root}")
        return
    try:
        _mkdir_exact(plan.claim_root)
    except FileExistsError:
        _reject_symlink(plan.claim_root, "claim root")
        if not plan.claim_root.is_dir():
            raise ClaimValidationError(
                f"claim root raced with a non-directory: {plan.claim_root}"
            )


def _active_reservation_path(claim_dir: Path) -> Path:
    return claim_dir / "active_submission"


def _create_active_reservation(
    claim_dir: Path,
    *,
    claim_uuid: str,
    attempt_number: int,
    attempt_uuid: str,
    submission_kind: str,
    reservation_token: str,
) -> None:
    active = _active_reservation_path(claim_dir)
    try:
        _mkdir_exact(active)
    except FileExistsError as exc:
        _reject_symlink(active, "active submission reservation")
        raise ClaimBlockedError("an active submission reservation already exists") from exc
    _atomic_write_json(
        active / "reservation.json",
        {
            "claim_uuid": claim_uuid,
            "attempt_number": attempt_number,
            "attempt_uuid": attempt_uuid,
            "submission_kind": submission_kind,
            "reservation_token_sha256": hashlib.sha256(
                reservation_token.encode("utf-8")
            ).hexdigest(),
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
            "pid": os.getpid(),
            "created_at_utc": _utc_now(),
        },
    )


def _archive_current_submission_reservation(
    claim_dir: Path,
    claim_uuid: str,
    state: Mapping[str, Any],
) -> None:
    active = _active_reservation_path(claim_dir)
    if not _lexists(active):
        return
    _reject_symlink(active, "active submission reservation")
    attempt_uuid = _validate_uuid(
        state.get("current_attempt_uuid"), "current submission attempt UUID"
    )
    reservation = _read_json(active / "reservation.json")
    if reservation.get("claim_uuid") != claim_uuid or reservation.get(
        "attempt_uuid"
    ) != attempt_uuid:
        raise ClaimValidationError(
            "active submission reservation does not match the current claim attempt"
        )
    _archive_active_directory(
        claim_dir, "active_submission", attempt_uuid
    )


def _archive_active_directory(claim_dir: Path, name: str, label: str) -> None:
    _validate_claim_container(claim_dir)
    if name not in {"active_submission", "active_owner"}:
        raise ClaimValidationError("active archive type is invalid")
    safe_label = _validate_uuid(label, "active archive label")
    active = claim_dir / name
    if not _lexists(active):
        return
    _reject_symlink(active, name)
    archive_root = claim_dir / f"{name}_history"
    _reject_symlink(archive_root, f"{name} history")
    if not _lexists(archive_root):
        _mkdir_exact(archive_root)
    if not archive_root.is_dir():
        raise ClaimValidationError(f"archive root is not a directory: {archive_root}")
    destination = archive_root / safe_label
    if destination.parent != archive_root:
        raise ClaimValidationError("active archive destination escapes its root")
    if _lexists(destination):
        raise ClaimValidationError(f"archive destination already exists: {destination}")
    os.replace(active, destination)
    _fsync_directory(claim_dir)


def _verify_initial_output_unclaimed(handle: ClaimHandle) -> None:
    """Bind the new claim only when no canonical output ownership exists."""

    plan = handle.plan
    conflict: str | None = None
    with _transition_lock(plan.claim_dir):
        try:
            _reject_symlink(plan.expected_output_path, "expected output")
            _reject_symlink(plan.incomplete_output_path, "incomplete output")
            if _lexists(plan.expected_output_path):
                conflict = "finalized_output_already_exists"
            elif _lexists(plan.incomplete_output_path):
                conflict = "incomplete_output_already_exists"
            elif _active_owner_record(plan.claim_dir) is not None:
                conflict = "active_output_owner_already_exists"
        except SubmissionClaimError as exc:
            conflict = f"unsafe_output_path_{type(exc).__name__.lower()}"
        if conflict is not None:
            _transition_locked(
                plan.claim_dir,
                plan.claim_uuid,
                ClaimState.BLOCKED_CONFLICT,
                event_type="INITIAL_OUTPUT_CONFLICT",
                updates={"output_conflict_reason": conflict},
                details={"reason": conflict},
            )
    if conflict is not None:
        _archive_active_directory(
            plan.claim_dir, "active_submission", handle.attempt_uuid
        )
        raise ClaimConflictError(
            f"canonical run output is already owned or unsafe: {conflict}"
        )


def acquire_initial_claim(plan: ClaimPlan) -> ClaimHandle:
    """Atomically reserve one canonical run ID for its exact specification."""

    if not plan.complete_for_acquisition:
        raise ClaimValidationError("initial acquisition requires all exact hashes")
    _ensure_claim_root(plan)
    _reject_symlink(plan.claim_dir, "claim directory")
    try:
        _mkdir_exact(plan.claim_dir)
    except FileExistsError as exc:
        _reject_symlink(plan.claim_dir, "claim directory")
        try:
            identity, _state = _load_claim(plan.claim_dir)
        except SubmissionClaimError as integrity_exc:
            raise ClaimBlockedError(
                f"an incomplete or invalid claim blocks run {plan.run_id}"
            ) from integrity_exc
        if (
            identity.get("base_resolved_spec_sha256")
            != plan.base_resolved_spec_sha256
            or identity.get("execution_resolved_spec_sha256")
            != plan.execution_resolved_spec_sha256
        ):
            raise ClaimConflictError(
                f"run_id {plan.run_id} is claimed by a different resolved specification"
            ) from exc
        raise ClaimBlockedError(
            f"run_id {plan.run_id} already has claim {identity.get('claim_uuid')}"
        ) from exc

    try:
        _mkdir_exact(plan.claim_dir / "history")
        attempt_number = 1
        attempt_uuid = _attempt_uuid(plan.claim_uuid, attempt_number)
        reservation_token = secrets.token_hex(32)
        _create_active_reservation(
            plan.claim_dir,
            claim_uuid=plan.claim_uuid,
            attempt_number=attempt_number,
            attempt_uuid=attempt_uuid,
            submission_kind="initial",
            reservation_token=reservation_token,
        )
        created = _utc_now()
        identity = {
            "schema_version": CLAIM_SCHEMA,
            "claim_uuid": plan.claim_uuid,
            "canonical_run_id": plan.run_id,
            "canonical_thermal_run_root": str(plan.run_root),
            "canonical_expected_output_path": str(plan.expected_output_path),
            "canonical_incomplete_output_path": str(plan.incomplete_output_path),
            "base_resolved_spec_sha256": plan.base_resolved_spec_sha256,
            "execution_resolved_spec_sha256": plan.execution_resolved_spec_sha256,
            "frozen_config_sha256": plan.frozen_config_sha256,
            "thermal_implementation_inventory_sha256": (
                plan.thermal_implementation_inventory_sha256
            ),
            "job_script_sha256": plan.job_script_sha256,
            "run_class": plan.run_class,
            "scientific_eligible": plan.scientific_eligible,
            "reuse_eligible": plan.reuse_eligible,
            "scheduler_job_name": plan.scheduler_job_name,
            "submitting_hostname": socket.gethostname(),
            "submitting_user": getpass.getuser(),
            "submitting_pid": os.getpid(),
            "created_at_utc": created,
        }
        _atomic_write_json(plan.claim_dir / "identity.json", identity)
        attempt = {
            "attempt_number": attempt_number,
            "attempt_uuid": attempt_uuid,
            "submission_kind": "initial",
            "reserved_at_utc": created,
            "qsub_invoked": False,
            "scheduler_job_id": None,
            "result": "RESERVED",
        }
        state = {
            "schema_version": STATE_SCHEMA,
            "claim_uuid": plan.claim_uuid,
            "state": ClaimState.RESERVED.value,
            "created_at_utc": created,
            "updated_at_utc": created,
            "submission_attempt_number": attempt_number,
            "current_attempt_uuid": attempt_uuid,
            "scheduler_job_id": None,
            "scheduler_job_name": plan.scheduler_job_name,
            "pending_owner_uuid": None,
            "active_owner_uuid": None,
            "attempts": [attempt],
            "history_sequence": 0,
            "history_head": None,
            "history_head_sha256": None,
        }
        _commit_state(
            plan.claim_dir,
            state,
            previous_state=None,
            event_type="CLAIM_RESERVED",
            details={
                "submission_kind": "initial",
                "attempt_uuid": attempt_uuid,
                "identity_sha256": sha256_file(plan.claim_dir / "identity.json"),
            },
        )
        handle = ClaimHandle(
            plan,
            attempt_number,
            attempt_uuid,
            "initial",
            reservation_token,
        )
        _verify_initial_output_unclaimed(handle)
        return handle
    except BaseException:
        # The atomically created directory remains as a blocking tombstone.
        raise


def _sanitize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    text = _ANSI_RE.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    return "".join(
        character
        for character in text
        if character in "\n\t" or (ord(character) >= 32 and ord(character) != 127)
    )


def _sanitized_command(command: Sequence[str]) -> list[str]:
    result: list[str] = []
    for token in command:
        text = str(token)
        marker = "THERMAL_JOB_PAYLOAD_B64="
        if marker in text:
            prefix, payload = text.split(marker, 1)
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            text = f"{prefix}{marker}<redacted-sha256:{digest}>"
        result.append(_sanitize_text(text))
    return result


def scheduler_job_name_from_qsub_command(command: Sequence[str]) -> str | None:
    """Return the single explicit SGE ``-N`` identity, or fail closed."""

    tokens = list(map(str, command))
    names: list[str] = []
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "-N":
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("-"):
                raise ClaimValidationError("qsub -N requires an explicit job name")
            names.append(tokens[index + 1])
            index += 2
            continue
        if token.startswith("-N") and len(token) > 2:
            names.append(token[2:])
        index += 1
    if not names:
        return None
    validated = [_validate_scheduler_job_name(name) for name in names]
    if len(validated) != 1:
        raise ClaimValidationError("qsub command must contain exactly one -N job name")
    return validated[0]


def parse_sge_job_id(stdout: str) -> str | None:
    match = _SGE_JOB_RE.fullmatch(_sanitize_text(stdout))
    return match.group(1) if match is not None else None


def _update_attempt(
    state: dict[str, Any], attempt_uuid: str, updates: Mapping[str, Any]
) -> list[dict[str, Any]]:
    attempts = state.get("attempts")
    if not isinstance(attempts, list):
        raise ClaimValidationError("claim state attempts are malformed")
    copied: list[dict[str, Any]] = []
    found = False
    for raw in attempts:
        if not isinstance(raw, dict):
            raise ClaimValidationError("claim attempt is malformed")
        item = dict(raw)
        if item.get("attempt_uuid") == attempt_uuid:
            if found:
                raise ClaimValidationError("duplicate claim attempt UUID")
            item.update(dict(updates))
            found = True
        copied.append(item)
    if not found:
        raise ClaimValidationError("current claim attempt UUID is missing")
    return copied


def _verify_submission_reservation(
    handle: ClaimHandle,
    state: Mapping[str, Any],
    *,
    require_invoked: bool,
) -> dict[str, Any]:
    if ClaimState(str(state.get("state"))) is not ClaimState.RESERVED:
        raise ClaimBlockedError("submission operation is allowed only from RESERVED")
    if state.get("current_attempt_uuid") != handle.attempt_uuid:
        raise ClaimBlockedError("submission attempt is no longer current")
    if state.get("submission_attempt_number") != handle.attempt_number:
        raise ClaimValidationError("submission attempt number mismatch")
    attempts = state.get("attempts")
    if not isinstance(attempts, list):
        raise ClaimValidationError("claim state attempts are malformed")
    matches = [
        item
        for item in attempts
        if isinstance(item, dict) and item.get("attempt_uuid") == handle.attempt_uuid
    ]
    if len(matches) != 1:
        raise ClaimValidationError("current submission attempt is missing or duplicated")
    invoked = matches[0].get("qsub_invoked")
    if invoked is not require_invoked:
        if invoked is True:
            raise ClaimBlockedError("qsub was already invoked for this attempt")
        raise ClaimValidationError("qsub invocation state is inconsistent")
    _verify_active_reservation_file(handle)
    return matches[0]


def _verify_active_reservation_file(handle: ClaimHandle) -> dict[str, Any]:
    reservation_path = _active_reservation_path(handle.plan.claim_dir)
    _reject_symlink(reservation_path, "active submission reservation")
    if not reservation_path.is_dir():
        raise ClaimBlockedError("active submission reservation is missing")
    reservation = _read_json(reservation_path / "reservation.json")
    expected = {
        "claim_uuid": handle.plan.claim_uuid,
        "attempt_number": handle.attempt_number,
        "attempt_uuid": handle.attempt_uuid,
        "submission_kind": handle.submission_kind,
        "reservation_token_sha256": hashlib.sha256(
            handle.reservation_token.encode("utf-8")
        ).hexdigest(),
    }
    for field, value in expected.items():
        if reservation.get(field) != value:
            raise ClaimValidationError(
                f"active submission reservation mismatch: {field}"
            )
    return reservation


def _read_resumable_output_marker(
    plan: ClaimPlan,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    _reject_symlink(plan.incomplete_output_path, "incomplete output")
    if not plan.incomplete_output_path.is_dir():
        raise ClaimBlockedError("resume requires an existing .incomplete run")
    marker = _read_json(plan.incomplete_output_path / OWNER_MARKER_NAME)
    expected_marker = {
        "schema_version": OWNER_SCHEMA,
        "claim_uuid": plan.claim_uuid,
        "canonical_run_id": plan.run_id,
        "canonical_output_root": str(plan.run_root),
        "canonical_expected_output_path": str(plan.expected_output_path),
        "canonical_incomplete_output_path": str(plan.incomplete_output_path),
        "execution_resolved_spec_sha256": identity.get(
            "execution_resolved_spec_sha256"
        ),
        "run_class": plan.run_class,
        "scientific_eligible": plan.scientific_eligible,
        "reuse_eligible": plan.reuse_eligible,
    }
    for field_name, expected_value in expected_marker.items():
        if marker.get(field_name) != expected_value:
            raise ClaimValidationError(
                f"resumable output owner marker mismatch: {field_name}"
            )
    _validate_uuid(marker.get("owner_uuid"), "resumable output owner UUID")
    _validate_uuid(marker.get("attempt_uuid"), "resumable submission attempt UUID")
    return marker


def _record_qsub_invocation(handle: ClaimHandle, command: Sequence[str]) -> None:
    _validate_plan_geometry(handle.plan)
    command_job_name = scheduler_job_name_from_qsub_command(command)
    if command_job_name != handle.plan.scheduler_job_name:
        raise ClaimValidationError(
            "qsub scheduler job name differs from the claim plan"
        )
    with _transition_lock(handle.plan.claim_dir):
        identity, state = _load_claim(handle.plan.claim_dir)
        if identity.get("claim_uuid") != handle.plan.claim_uuid:
            raise ClaimValidationError("claim UUID mismatch before qsub")
        _verify_submission_reservation(handle, state, require_invoked=False)
        attempts = _update_attempt(
            state,
            handle.attempt_uuid,
            {
                "qsub_invoked": True,
                "qsub_invoked_at_utc": _utc_now(),
                "sanitized_command": _sanitized_command(command),
                "command_sha256": canonical_sha256(list(map(str, command))),
                "scheduler_job_name": command_job_name,
                "result": "INVOKED_OUTCOME_PENDING",
            },
        )
        _transition_locked(
            handle.plan.claim_dir,
            handle.plan.claim_uuid,
            ClaimState.RESERVED,
            event_type="QSUB_INVOCATION_RECORDED",
            updates={
                "attempts": attempts,
                "scheduler_job_name": command_job_name,
            },
            details={"attempt_uuid": handle.attempt_uuid},
            allow_same=True,
        )


def _record_submission_result(
    handle: ClaimHandle,
    *,
    target: ClaimState,
    returncode: int | None,
    stdout: str,
    stderr: str,
    scheduler_job_id: str | None,
    result: str,
    event_type: str,
) -> dict[str, Any]:
    with _transition_lock(handle.plan.claim_dir, wait_seconds=30.0):
        identity, state = _load_claim(handle.plan.claim_dir)
        if identity.get("claim_uuid") != handle.plan.claim_uuid:
            raise ClaimValidationError("claim UUID mismatch after qsub")
        _verify_submission_reservation(handle, state, require_invoked=True)
        output_conflict: str | None = None
        if target in {ClaimState.SUBMITTED, ClaimState.SUBMISSION_FAILED}:
            try:
                _reject_symlink(handle.plan.expected_output_path, "expected output")
                _reject_symlink(
                    handle.plan.incomplete_output_path, "incomplete output"
                )
                if _lexists(handle.plan.expected_output_path):
                    output_conflict = "finalized_output_appeared_during_qsub"
                elif handle.submission_kind == "resume":
                    _read_resumable_output_marker(handle.plan, identity)
                elif _lexists(handle.plan.incomplete_output_path):
                    output_conflict = "incomplete_output_appeared_during_qsub"
                if (
                    output_conflict is None
                    and _active_owner_record(handle.plan.claim_dir) is not None
                ):
                    output_conflict = "active_output_owner_appeared_during_qsub"
            except SubmissionClaimError:
                output_conflict = "unsafe_output_evidence_during_qsub"
        if output_conflict is not None:
            target = ClaimState.SUBMISSION_UNKNOWN
            result = "CONTRADICTORY_SCHEDULER_AND_OUTPUT_EVIDENCE"
            event_type = "SUBMISSION_OUTCOME_UNKNOWN"
        attempts = _update_attempt(
            state,
            handle.attempt_uuid,
            {
                "completed_at_utc": _utc_now(),
                "returncode": returncode,
                "stdout": _sanitize_text(stdout),
                "stderr": _sanitize_text(stderr),
                "scheduler_job_id": scheduler_job_id,
                "result": result,
                "output_conflict": output_conflict,
            },
        )
        updated = _transition_locked(
            handle.plan.claim_dir,
            handle.plan.claim_uuid,
            target,
            event_type=event_type,
            updates={
                "attempts": attempts,
                "scheduler_job_id": scheduler_job_id,
                "output_conflict": output_conflict,
            },
            details={
                "attempt_uuid": handle.attempt_uuid,
                "returncode": returncode,
                "scheduler_job_id": scheduler_job_id,
            },
        )
        if target is ClaimState.SUBMISSION_FAILED:
            _verify_active_reservation_file(handle)
            late_conflict: str | None = None
            try:
                _reject_symlink(handle.plan.expected_output_path, "expected output")
                _reject_symlink(
                    handle.plan.incomplete_output_path, "incomplete output"
                )
                if _lexists(handle.plan.expected_output_path):
                    late_conflict = "finalized_output_before_failure_release"
                elif handle.submission_kind == "resume":
                    _read_resumable_output_marker(handle.plan, identity)
                elif _lexists(handle.plan.incomplete_output_path):
                    late_conflict = "incomplete_output_before_failure_release"
                if (
                    late_conflict is None
                    and _active_owner_record(handle.plan.claim_dir) is not None
                ):
                    late_conflict = "active_owner_before_failure_release"
            except SubmissionClaimError:
                late_conflict = "unsafe_output_before_failure_release"
            if late_conflict is None and handle.submission_kind == "initial":
                updated = _transition_locked(
                    handle.plan.claim_dir,
                    handle.plan.claim_uuid,
                    ClaimState.RELEASED,
                    event_type="FAILED_RESERVATION_RELEASED",
                    updates={"release_reason": "definite_qsub_failure"},
                    details={
                        "owner_safe": True,
                        "reservation_token_verified": True,
                        "scheduler_acceptance_evidence": False,
                    },
                )
                target = ClaimState.RELEASED
            elif late_conflict is not None:
                late_attempts = _update_attempt(
                    updated,
                    handle.attempt_uuid,
                    {
                        "result": "CONTRADICTORY_AFTER_DEFINITE_FAILURE",
                        "output_conflict": late_conflict,
                    },
                )
                updated = _transition_locked(
                    handle.plan.claim_dir,
                    handle.plan.claim_uuid,
                    ClaimState.SUBMISSION_UNKNOWN,
                    event_type="SUBMISSION_OUTCOME_UNKNOWN",
                    updates={
                        "attempts": late_attempts,
                        "output_conflict": late_conflict,
                    },
                    details={
                        "attempt_uuid": handle.attempt_uuid,
                        "output_conflict": late_conflict,
                    },
                )
                target = ClaimState.SUBMISSION_UNKNOWN
    if target in {
        ClaimState.SUBMITTED,
        ClaimState.SUBMISSION_FAILED,
        ClaimState.RELEASED,
    }:
        _archive_active_directory(
            handle.plan.claim_dir, "active_submission", handle.attempt_uuid
        )
    return updated


QsubRunner = Callable[..., subprocess.CompletedProcess[Any]]


def invoke_qsub_once(
    handle: ClaimHandle,
    command: Sequence[str],
    *,
    runner: QsubRunner | None = None,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Invoke the injected qsub runner exactly once and classify its outcome."""

    if not command or str(command[0]) != "qsub":
        raise ClaimValidationError("submission command must begin with qsub")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise ClaimValidationError("qsub timeout must be positive and finite")
    _record_qsub_invocation(handle, command)
    selected_runner = runner or subprocess.run
    try:
        completed = selected_runner(
            list(map(str, command)),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return _record_submission_result(
            handle,
            target=ClaimState.SUBMISSION_UNKNOWN,
            returncode=None,
            stdout=_sanitize_text(exc.stdout),
            stderr=_sanitize_text(exc.stderr),
            scheduler_job_id=None,
            result="TIMEOUT_AMBIGUOUS",
            event_type="SUBMISSION_OUTCOME_UNKNOWN",
        )
    except FileNotFoundError as exc:
        failed = _record_submission_result(
            handle,
            target=ClaimState.SUBMISSION_FAILED,
            returncode=127,
            stdout="",
            stderr=str(exc),
            scheduler_job_id=None,
            result="QSUB_EXECUTABLE_NOT_FOUND",
            event_type="SUBMISSION_DEFINITELY_FAILED",
        )
        return failed
    except OSError as exc:
        return _record_submission_result(
            handle,
            target=ClaimState.SUBMISSION_UNKNOWN,
            returncode=None,
            stdout="",
            stderr=str(exc),
            scheduler_job_id=None,
            result="TRANSPORT_OR_OS_ERROR_AMBIGUOUS",
            event_type="SUBMISSION_OUTCOME_UNKNOWN",
        )
    except Exception as exc:
        return _record_submission_result(
            handle,
            target=ClaimState.SUBMISSION_UNKNOWN,
            returncode=None,
            stdout="",
            stderr=f"{type(exc).__name__}: {exc}",
            scheduler_job_id=None,
            result="RUNNER_EXCEPTION_AMBIGUOUS",
            event_type="SUBMISSION_OUTCOME_UNKNOWN",
        )

    try:
        stdout = _sanitize_text(completed.stdout)
        stderr = _sanitize_text(completed.stderr)
        raw_returncode = completed.returncode
        if isinstance(raw_returncode, bool):
            raise ValueError("boolean return code")
        returncode = int(raw_returncode)
    except (AttributeError, TypeError, ValueError) as exc:
        return _record_submission_result(
            handle,
            target=ClaimState.SUBMISSION_UNKNOWN,
            returncode=None,
            stdout="",
            stderr=f"malformed qsub runner result: {type(exc).__name__}",
            scheduler_job_id=None,
            result="RUNNER_RESULT_MALFORMED",
            event_type="SUBMISSION_OUTCOME_UNKNOWN",
        )
    stdout_job_id = parse_sge_job_id(stdout)
    stderr_job_id = parse_sge_job_id(stderr)
    observed_ids = {
        value for value in (stdout_job_id, stderr_job_id) if value is not None
    }
    observed_job_id = next(iter(observed_ids)) if len(observed_ids) == 1 else None
    success_like = (
        "submitted" in stdout.lower()
        or "submitted" in stderr.lower()
        or bool(observed_ids)
    )
    if returncode == 0 and stdout_job_id is not None and not stderr.strip():
        return _record_submission_result(
            handle,
            target=ClaimState.SUBMITTED,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            scheduler_job_id=stdout_job_id,
            result="SUBMITTED",
            event_type="SUBMISSION_ACCEPTED",
        )
    if returncode != 0 and observed_job_id is None and not success_like:
        failed = _record_submission_result(
            handle,
            target=ClaimState.SUBMISSION_FAILED,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            scheduler_job_id=None,
            result="DEFINITE_NONZERO_FAILURE",
            event_type="SUBMISSION_DEFINITELY_FAILED",
        )
        return failed
    return _record_submission_result(
        handle,
        target=ClaimState.SUBMISSION_UNKNOWN,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        scheduler_job_id=observed_job_id,
        result="AMBIGUOUS_OR_UNPARSABLE_RESULT",
        event_type="SUBMISSION_OUTCOME_UNKNOWN",
    )


def classify_scheduler_evidence(
    *,
    command_available: bool,
    returncode: int | None,
    stdout: object,
    stderr: object,
    expected_job_id: str | None,
    expected_job_name: str | None = None,
    observed_at_utc: str | None = None,
) -> SchedulerObservation:
    """Classify SGE-style qstat output conservatively without running qstat."""

    clean_stdout = _sanitize_text(stdout)
    clean_stderr = _sanitize_text(stderr)
    parsed_observed: datetime | None = None
    if observed_at_utc is None:
        observed = _utc_now()
    else:
        if not isinstance(observed_at_utc, str) or not observed_at_utc.endswith("Z"):
            raise ClaimValidationError(
                "scheduler observed_at_utc must be an explicit UTC timestamp"
            )
        try:
            parsed_observed = datetime.fromisoformat(
                observed_at_utc.removesuffix("Z") + "+00:00"
            )
        except ValueError as exc:
            raise ClaimValidationError(
                "scheduler observed_at_utc is malformed"
            ) from exc
        if parsed_observed.tzinfo is None:
            raise ClaimValidationError("scheduler observed_at_utc lacks a timezone")
        observed = observed_at_utc
    if parsed_observed is not None:
        age_seconds = (
            datetime.now(timezone.utc) - parsed_observed.astimezone(timezone.utc)
        ).total_seconds()
        if (
            age_seconds > SCHEDULER_EVIDENCE_MAX_AGE_SECONDS
            or age_seconds < -SCHEDULER_EVIDENCE_MAX_FUTURE_SKEW_SECONDS
        ):
            return SchedulerObservation(
                SchedulerEvidence.UNKNOWN,
                observed,
                bool(command_available),
                returncode if isinstance(returncode, int) else None,
                clean_stdout,
                clean_stderr,
                (),
                "scheduler_evidence_stale_or_future",
            )
    if type(command_available) is not bool:
        return SchedulerObservation(
            SchedulerEvidence.UNKNOWN,
            observed,
            False,
            None,
            clean_stdout,
            clean_stderr,
            (),
            "scheduler_command_availability_malformed",
        )
    if not command_available:
        return SchedulerObservation(
            SchedulerEvidence.UNKNOWN,
            observed,
            False,
            returncode,
            clean_stdout,
            clean_stderr,
            (),
            "scheduler_command_unavailable",
        )
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        return SchedulerObservation(
            SchedulerEvidence.UNKNOWN,
            observed,
            True,
            None,
            clean_stdout,
            clean_stderr,
            (),
            "scheduler_returncode_malformed",
        )
    if returncode != 0:
        return SchedulerObservation(
            SchedulerEvidence.UNKNOWN,
            observed,
            True,
            returncode,
            clean_stdout,
            clean_stderr,
            (),
            "scheduler_command_nonzero",
        )
    if clean_stderr.strip():
        return SchedulerObservation(
            SchedulerEvidence.UNKNOWN,
            observed,
            True,
            returncode,
            clean_stdout,
            clean_stderr,
            (),
            "scheduler_stderr_nonempty",
        )
    lines = [line.strip() for line in clean_stdout.splitlines() if line.strip()]
    required_header = ["job-id", "prior", "name", "user", "state"]
    header_valid = bool(
        len(lines) >= 2
        and lines[0].lower().split()[: len(required_header)] == required_header
        and len(lines[1].replace(" ", "")) >= 5
        and set(lines[1].replace(" ", "")) == {"-"}
    )
    rows: list[tuple[str, str]] = []
    malformed = not header_valid
    if header_valid:
        for line in lines[2:]:
            fields = line.split()
            if fields and fields[0].isdigit() and len(fields) >= 5:
                rows.append((fields[0], fields[2]))
            else:
                malformed = True
    if malformed:
        return SchedulerObservation(
            SchedulerEvidence.UNKNOWN,
            observed,
            True,
            returncode,
            clean_stdout,
            clean_stderr,
            (),
            "scheduler_output_unparsable",
        )
    expected_id = str(expected_job_id) if expected_job_id is not None else None
    if expected_id is not None and not expected_id.isdigit():
        return SchedulerObservation(
            SchedulerEvidence.UNKNOWN,
            observed,
            True,
            returncode,
            clean_stdout,
            clean_stderr,
            (),
            "scheduler_job_id_malformed",
        )
    if expected_job_name is not None:
        _validate_scheduler_job_name(expected_job_name)
    by_id = [row for row in rows if expected_id is not None and row[0] == expected_id]
    def _name_may_match(observed_name: str) -> bool:
        if expected_job_name is None:
            return False
        return observed_name == expected_job_name or expected_job_name.startswith(
            observed_name
        )

    by_name = [row for row in rows if _name_may_match(row[1])]
    if expected_id is not None:
        contradictory = [row for row in by_name if row[0] != expected_id]
        id_name_mismatch = bool(
            expected_job_name is not None
            and by_id
            and any(not _name_may_match(row[1]) for row in by_id)
        )
        if len(by_id) > 1 or contradictory or id_name_mismatch:
            status = SchedulerEvidence.CONFLICT
            matches = tuple(row[0] for row in by_id + contradictory)
            reason = "contradictory_or_multiple_scheduler_jobs"
        elif len(by_id) == 1:
            status = SchedulerEvidence.ACTIVE
            matches = (by_id[0][0],)
            reason = "expected_scheduler_job_active"
        else:
            status = SchedulerEvidence.ABSENT
            matches = ()
            reason = "expected_scheduler_job_absent"
    elif expected_job_name is not None:
        if len(by_name) > 1:
            status = SchedulerEvidence.CONFLICT
            matches = tuple(row[0] for row in by_name)
            reason = "multiple_scheduler_jobs_match_name"
        elif len(by_name) == 1:
            status = SchedulerEvidence.ACTIVE
            matches = (by_name[0][0],)
            reason = "scheduler_job_name_active"
        else:
            status = SchedulerEvidence.ABSENT
            matches = ()
            reason = "scheduler_job_name_absent"
    else:
        status = SchedulerEvidence.UNKNOWN
        matches = ()
        reason = "scheduler_identity_not_supplied"
    return SchedulerObservation(
        status,
        observed,
        True,
        returncode,
        clean_stdout,
        clean_stderr,
        matches,
        reason,
        expected_id,
        expected_job_name,
    )


def _scheduler_absence_is_current(
    scheduler: SchedulerObservation,
    identity: Mapping[str, Any],
    state: Mapping[str, Any],
) -> bool:
    current_job_id = state.get("scheduler_job_id")
    expected_current_id = (
        str(current_job_id) if current_job_id is not None else None
    )
    current_job_name = state.get(
        "scheduler_job_name", identity.get("scheduler_job_name")
    )
    current_state = ClaimState(str(state.get("state")))
    unsubmitted_reservation = bool(
        current_state is ClaimState.RESERVED
        and not _submission_outcome_is_pending(state)
    )
    try:
        reclassified = classify_scheduler_evidence(
            command_available=scheduler.command_available,
            returncode=scheduler.returncode,
            stdout=scheduler.stdout,
            stderr=scheduler.stderr,
            expected_job_id=expected_current_id,
            expected_job_name=current_job_name,
            observed_at_utc=scheduler.observed_at_utc,
        )
    except (AttributeError, TypeError, ValueError, SubmissionClaimError):
        return False
    if reclassified.status is not SchedulerEvidence.ABSENT:
        return False
    if scheduler != reclassified:
        return False
    if scheduler.expected_job_id != expected_current_id:
        return False
    if not unsubmitted_reservation and scheduler.expected_job_name != current_job_name:
        return False
    try:
        observed = datetime.fromisoformat(
            scheduler.observed_at_utc.removesuffix("Z") + "+00:00"
        )
        state_updated = datetime.fromisoformat(
            str(state.get("updated_at_utc", "")).removesuffix("Z") + "+00:00"
        )
    except (AttributeError, ValueError):
        return False
    if observed < state_updated:
        return False
    age = (datetime.now(timezone.utc) - observed.astimezone(timezone.utc)).total_seconds()
    return (
        -SCHEDULER_EVIDENCE_MAX_FUTURE_SKEW_SECONDS
        <= age
        <= SCHEDULER_EVIDENCE_MAX_AGE_SECONDS
    )


def _verify_job_identity(
    identity: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    claim_uuid: str,
    run_id: str,
    run_root: Path,
    execution_resolved_spec_sha256: str,
    run_class: str,
    scientific_eligible: bool,
    reuse_eligible: bool,
    attempt_number: int,
    attempt_uuid: str,
    submission_kind: str,
    scheduler_job_id: str | None,
) -> tuple[int, bool]:
    expected = {
        "claim_uuid": claim_uuid,
        "canonical_run_id": run_id,
        "canonical_thermal_run_root": str(run_root),
        "canonical_expected_output_path": str(run_root / run_id),
        "canonical_incomplete_output_path": str(
            run_root / f"{run_id}.incomplete"
        ),
        "execution_resolved_spec_sha256": execution_resolved_spec_sha256,
        "run_class": run_class,
        "scientific_eligible": scientific_eligible,
        "reuse_eligible": reuse_eligible,
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ClaimValidationError(f"job authorization mismatch: {key}")
    if ClaimState(str(state.get("state"))) is not ClaimState.SUBMITTED:
        raise ClaimBlockedError("job startup requires claim state SUBMITTED")
    if state.get("current_attempt_uuid") != attempt_uuid:
        raise ClaimValidationError("job submission attempt UUID mismatch")
    if state.get("submission_attempt_number") != attempt_number:
        raise ClaimValidationError("job submission attempt number mismatch")
    claimed_job_id = state.get("scheduler_job_id")
    if claimed_job_id is not None:
        if scheduler_job_id is None:
            raise ClaimValidationError(
                "scheduler job ID is required for this submitted claim"
            )
        if str(claimed_job_id) != str(scheduler_job_id):
            raise ClaimValidationError("scheduler job ID does not match the claim")
    attempts = state.get("attempts")
    if not isinstance(attempts, list):
        raise ClaimValidationError("claim attempts are malformed")
    if any(not isinstance(item, dict) for item in attempts):
        raise ClaimValidationError("claim attempt entry is malformed")
    matches = [item for item in attempts if item.get("attempt_uuid") == attempt_uuid]
    if len(matches) != 1:
        raise ClaimValidationError("job attempt is missing or duplicated")
    receipt = _validated_submission_receipt(state)
    if receipt is not matches[0]:
        raise ClaimValidationError("job attempt differs from the accepted qsub receipt")
    recorded_number = int(receipt.get("attempt_number", 0))
    recorded_kind = receipt.get("submission_kind")
    if recorded_number != attempt_number:
        raise ClaimValidationError("job attempt record number mismatch")
    if submission_kind not in {"initial", "resume"} or recorded_kind != submission_kind:
        raise ClaimValidationError("job submission kind mismatch")
    return recorded_number, recorded_kind == "resume"


def _active_owner_record(claim_dir: Path) -> dict[str, Any] | None:
    active = claim_dir / "active_owner"
    if not _lexists(active):
        return None
    _reject_symlink(active, "active output owner")
    if not active.is_dir():
        raise ClaimValidationError("active output owner is not a directory")
    return _read_json(active / "owner.json")


def authorize_job_start(
    *,
    run_root: str | Path,
    run_id: str,
    claim_uuid: str,
    execution_resolved_spec_sha256: str,
    run_class: str,
    scientific_eligible: bool,
    reuse_eligible: bool,
    attempt_number: int,
    attempt_uuid: str,
    submission_kind: str,
    scheduler_job_id: str | None,
    submission_wait_seconds: float = 30.0,
) -> OutputOwnership:
    """Validate one submitted job and atomically acquire its output owner."""

    root = canonical_run_root(run_root, must_exist=True)
    identifier = validate_run_id(run_id)
    spec_hash = _validate_sha256(
        execution_resolved_spec_sha256, "execution_resolved_spec_sha256"
    )
    claim_dir = root / CLAIM_ROOT_NAME / f"{identifier}.claim"
    _contained(claim_dir, root, "claim directory")
    if (
        isinstance(submission_wait_seconds, bool)
        or not isinstance(submission_wait_seconds, (int, float))
        or not math.isfinite(float(submission_wait_seconds))
        or submission_wait_seconds < 0
    ):
        raise ClaimValidationError(
            "submission wait must be finite and nonnegative"
        )
    deadline = time.monotonic() + submission_wait_seconds
    while True:
        _identity_snapshot, state_snapshot = _load_claim(claim_dir)
        snapshot_state = ClaimState(str(state_snapshot.get("state")))
        if snapshot_state is ClaimState.SUBMITTED:
            break
        if snapshot_state is not ClaimState.RESERVED or not _submission_outcome_is_pending(
            state_snapshot
        ):
            raise ClaimBlockedError(
                f"claim state {snapshot_state.value} does not authorize job startup"
            )
        if time.monotonic() >= deadline:
            raise ClaimBlockedError(
                "timed out waiting for the qsub receipt to authorize job startup"
            )
        time.sleep(0.1)
    remaining_wait = max(0.0, deadline - time.monotonic())
    with _transition_lock(claim_dir, wait_seconds=remaining_wait):
        identity, state = _load_claim(claim_dir)
        attempt_number, resume = _verify_job_identity(
            identity,
            state,
            claim_uuid=claim_uuid,
            run_id=identifier,
            run_root=root,
            execution_resolved_spec_sha256=spec_hash or "",
            run_class=run_class,
            scientific_eligible=scientific_eligible,
            reuse_eligible=reuse_eligible,
            attempt_number=attempt_number,
            attempt_uuid=attempt_uuid,
            submission_kind=submission_kind,
            scheduler_job_id=scheduler_job_id,
        )
        expected = root / identifier
        incomplete = root / f"{identifier}.incomplete"
        _reject_symlink(expected, "expected output")
        _reject_symlink(incomplete, "incomplete output")
        if _lexists(expected):
            raise ClaimConflictError(f"finalized output already exists: {expected}")
        if state.get("pending_owner_uuid") is not None:
            raise ClaimBlockedError(
                "a prior output-owner acquisition requires explicit recovery"
            )
        owner_dir = claim_dir / "active_owner"
        if _lexists(owner_dir):
            raise ClaimBlockedError("another job owns the active run")
        owner_lease = _acquire_output_owner_lease(
            claim_dir,
            wait_seconds=max(0.0, deadline - time.monotonic()),
        )
        owner_uuid = str(uuid.uuid4())
        owner_created = False
        try:
            _transition_locked(
                claim_dir,
                claim_uuid,
                ClaimState.SUBMITTED,
                event_type="OUTPUT_OWNER_ACQUISITION_STARTED",
                updates={"pending_owner_uuid": owner_uuid},
                details={
                    "attempt_uuid": attempt_uuid,
                    "owner_uuid": owner_uuid,
                },
                allow_same=True,
            )
            owner = {
                "schema_version": OWNER_SCHEMA,
                "claim_uuid": claim_uuid,
                "canonical_run_id": identifier,
                "canonical_output_root": str(root),
                "canonical_expected_output_path": str(expected),
                "canonical_incomplete_output_path": str(incomplete),
                "execution_resolved_spec_sha256": spec_hash,
                "run_class": run_class,
                "scientific_eligible": scientific_eligible,
                "reuse_eligible": reuse_eligible,
                "attempt_number": attempt_number,
                "attempt_uuid": attempt_uuid,
                "scheduler_job_id": scheduler_job_id,
                "owner_uuid": owner_uuid,
                "owner_hostname": socket.gethostname(),
                "owner_user": getpass.getuser(),
                "owner_pid": os.getpid(),
                "owner_lease_path": str(owner_lease.path),
                "resume": resume,
                "created_at_utc": _utc_now(),
            }
            try:
                _mkdir_exact(owner_dir)
                owner_created = True
            except FileExistsError as exc:
                raise ClaimBlockedError("another job owns the active run") from exc
            _atomic_write_json(owner_dir / "owner.json", owner)
            if resume:
                if not incomplete.is_dir():
                    raise ClaimValidationError(
                        "resume authorization requires an existing .incomplete run"
                    )
                previous_marker = _read_json(incomplete / OWNER_MARKER_NAME)
                for field in (
                    "claim_uuid",
                    "canonical_run_id",
                    "canonical_output_root",
                    "canonical_expected_output_path",
                    "canonical_incomplete_output_path",
                    "execution_resolved_spec_sha256",
                    "run_class",
                    "scientific_eligible",
                    "reuse_eligible",
                ):
                    if previous_marker.get(field) != owner[field]:
                        raise ClaimValidationError(
                            f"existing output ownership mismatch: {field}"
                        )
            else:
                if _lexists(incomplete):
                    raise ClaimConflictError(
                        f"initial output already exists: {incomplete}"
                    )
                _mkdir_exact(incomplete)
            _atomic_write_json(incomplete / OWNER_MARKER_NAME, owner)
            _transition_locked(
                claim_dir,
                claim_uuid,
                ClaimState.RUNNING,
                event_type="JOB_OUTPUT_OWNERSHIP_ACQUIRED",
                updates={
                    "active_owner_uuid": owner_uuid,
                    "pending_owner_uuid": None,
                },
                details={
                    "attempt_uuid": attempt_uuid,
                    "owner_uuid": owner_uuid,
                    "resume": resume,
                },
            )
        except BaseException:
            try:
                if owner_created and _lexists(owner_dir):
                    _archive_active_directory(
                        claim_dir, "active_owner", owner_uuid
                    )
            finally:
                _release_owner_lease(owner_lease)
            raise
    return OutputOwnership(
        claim_dir=claim_dir,
        claim_uuid=claim_uuid,
        run_root=root,
        run_id=identifier,
        expected_output_path=expected,
        incomplete_output_path=incomplete,
        execution_resolved_spec_sha256=spec_hash or "",
        attempt_number=attempt_number,
        attempt_uuid=attempt_uuid,
        owner_uuid=owner_uuid,
        resume=resume,
        owner_lease=owner_lease,
    )


def _verify_active_owner(ownership: OutputOwnership) -> dict[str, Any]:
    record = _active_owner_record(ownership.claim_dir)
    if record is None:
        raise ClaimBlockedError("active output owner is missing")
    expected = {
        "claim_uuid": ownership.claim_uuid,
        "canonical_run_id": ownership.run_id,
        "attempt_uuid": ownership.attempt_uuid,
        "owner_uuid": ownership.owner_uuid,
        "execution_resolved_spec_sha256": ownership.execution_resolved_spec_sha256,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ClaimValidationError(f"active output owner mismatch: {key}")
    return record


def mark_job_complete(
    ownership: OutputOwnership,
    *,
    final_manifest: str | Path | None = None,
) -> dict[str, Any]:
    _verify_output_owner_lease(ownership)
    try:
        with _transition_lock(ownership.claim_dir, wait_seconds=30.0):
            _verify_active_owner(ownership)
            if _lexists(ownership.incomplete_output_path):
                raise ClaimValidationError(
                    "cannot complete claim while .incomplete output still exists"
                )
            if not ownership.expected_output_path.is_dir():
                raise ClaimValidationError("final output directory is missing")
            manifest_hash = None
            if final_manifest is not None:
                manifest = Path(final_manifest).resolve()
                _contained(manifest, ownership.expected_output_path, "final manifest")
                manifest_hash = sha256_file(manifest)
            state = _transition_locked(
                ownership.claim_dir,
                ownership.claim_uuid,
                ClaimState.COMPLETE,
                event_type="JOB_COMPLETED",
                updates={
                    "active_owner_uuid": None,
                    "completed_output_path": str(ownership.expected_output_path),
                    "final_manifest_sha256": manifest_hash,
                },
                details={
                    "attempt_uuid": ownership.attempt_uuid,
                    "owner_uuid": ownership.owner_uuid,
                },
            )
        _archive_active_directory(
            ownership.claim_dir,
            "active_owner",
            ownership.owner_uuid,
        )
        return state
    finally:
        _release_output_owner_lease(ownership)


def mark_job_failed(
    ownership: OutputOwnership,
    *,
    reason: str,
    interrupted: bool = False,
) -> dict[str, Any]:
    machine_reason = _validate_reason(reason)
    target = ClaimState.INTERRUPTED if interrupted else ClaimState.FAILED
    _verify_output_owner_lease(ownership)
    try:
        with _transition_lock(ownership.claim_dir, wait_seconds=30.0):
            _verify_active_owner(ownership)
            state = _transition_locked(
                ownership.claim_dir,
                ownership.claim_uuid,
                target,
                event_type="JOB_INTERRUPTED" if interrupted else "JOB_FAILED",
                updates={
                    "active_owner_uuid": None,
                    "job_failure_reason": machine_reason,
                },
                details={
                    "attempt_uuid": ownership.attempt_uuid,
                    "owner_uuid": ownership.owner_uuid,
                    "reason": machine_reason,
                },
            )
        _archive_active_directory(
            ownership.claim_dir,
            "active_owner",
            ownership.owner_uuid,
        )
        return state
    finally:
        _release_output_owner_lease(ownership)


def revoke_orphaned_output_owner(
    plan: ClaimPlan,
    *,
    claim_uuid: str,
    owner_uuid: str,
    scheduler: SchedulerObservation,
    confirmation: str,
    reason: str,
) -> dict[str, Any]:
    """Explicitly revoke a dead job owner after fresh scheduler absence proof."""

    _validate_plan_geometry(plan)
    if confirmation != "REVOKE_ORPHANED_OUTPUT_OWNER":
        raise ClaimValidationError("owner-revocation confirmation token is invalid")
    exact_owner_uuid = _validate_uuid(owner_uuid, "output owner UUID")
    machine_reason = _validate_reason(reason)
    with _transition_lock(plan.claim_dir):
        identity, state = _load_claim(plan.claim_dir)
        if identity.get("claim_uuid") != claim_uuid:
            raise ClaimValidationError("owner-revocation claim UUID mismatch")
        current = ClaimState(str(state.get("state")))
        if current not in {
            ClaimState.SUBMITTED,
            ClaimState.RUNNING,
            ClaimState.INTERRUPTED,
            ClaimState.FAILED,
        }:
            raise ClaimBlockedError(
                f"claim state {current.value} cannot revoke an output owner"
            )
        attempts = state.get("attempts")
        current_attempt_uuid = state.get("current_attempt_uuid")
        if not isinstance(attempts, list) or any(
            not isinstance(item, dict) for item in attempts
        ):
            raise ClaimValidationError("claim attempts are malformed")
        current_attempts = [
            item
            for item in attempts
            if item.get("attempt_uuid") == current_attempt_uuid
        ]
        if len(current_attempts) != 1:
            raise ClaimValidationError("current claim attempt is missing or duplicated")
        submission_kind = current_attempts[0].get("submission_kind")
        if submission_kind not in {"initial", "resume"}:
            raise ClaimValidationError("current submission kind is invalid")
        if not _scheduler_absence_is_current(scheduler, identity, state):
            raise ClaimBlockedError(
                "owner revocation requires scheduler ABSENT evidence bound to the current claim revision"
            )
        if _lexists(plan.expected_output_path):
            raise ClaimBlockedError("a finalized run cannot revoke its output owner")
        partial_acquisition = bool(
            current in {ClaimState.SUBMITTED, ClaimState.INTERRUPTED}
            and state.get("pending_owner_uuid") == exact_owner_uuid
        )
        if current is ClaimState.SUBMITTED and not partial_acquisition:
            raise ClaimValidationError(
                "submitted claim does not bind the requested pending owner UUID"
            )
        active_owner_path = plan.claim_dir / "active_owner"
        owner: dict[str, Any] | None = None
        if _lexists(active_owner_path):
            _reject_symlink(active_owner_path, "active output owner")
            if not active_owner_path.is_dir():
                raise ClaimValidationError("active output owner is not a directory")
            owner_json = active_owner_path / "owner.json"
            if _lexists(owner_json):
                owner = _read_json(owner_json)
                owner_bindings = {
                    "schema_version": OWNER_SCHEMA,
                    "owner_uuid": exact_owner_uuid,
                    "claim_uuid": claim_uuid,
                    "canonical_run_id": plan.run_id,
                    "canonical_output_root": str(plan.run_root),
                    "canonical_expected_output_path": str(
                        plan.expected_output_path
                    ),
                    "canonical_incomplete_output_path": str(
                        plan.incomplete_output_path
                    ),
                    "execution_resolved_spec_sha256": identity.get(
                        "execution_resolved_spec_sha256"
                    ),
                    "run_class": plan.run_class,
                    "scientific_eligible": plan.scientific_eligible,
                    "reuse_eligible": plan.reuse_eligible,
                    "attempt_uuid": current_attempt_uuid,
                }
                for field_name, expected_value in owner_bindings.items():
                    if owner.get(field_name) != expected_value:
                        raise ClaimValidationError(
                            f"active output owner identity mismatch: {field_name}"
                        )
            elif not partial_acquisition:
                raise ClaimValidationError("active output owner metadata is missing")
        elif not partial_acquisition:
            raise ClaimValidationError("active output owner is missing")

        incomplete_present = _lexists(plan.incomplete_output_path)
        marker_present = False
        marker_matches_pending_owner = False
        if incomplete_present:
            _reject_symlink(plan.incomplete_output_path, "incomplete output")
            if not plan.incomplete_output_path.is_dir():
                raise ClaimValidationError("incomplete output is not a directory")
            marker_path = plan.incomplete_output_path / OWNER_MARKER_NAME
            marker_present = _lexists(marker_path)
            if marker_present:
                marker = _read_json(marker_path)
                marker_bindings = {
                    "schema_version": OWNER_SCHEMA,
                    "claim_uuid": claim_uuid,
                    "canonical_run_id": plan.run_id,
                    "canonical_output_root": str(plan.run_root),
                    "canonical_expected_output_path": str(
                        plan.expected_output_path
                    ),
                    "canonical_incomplete_output_path": str(
                        plan.incomplete_output_path
                    ),
                    "execution_resolved_spec_sha256": (
                        identity.get("execution_resolved_spec_sha256")
                    ),
                    "run_class": plan.run_class,
                    "scientific_eligible": plan.scientific_eligible,
                    "reuse_eligible": plan.reuse_eligible,
                }
                for field_name, expected_value in marker_bindings.items():
                    if marker.get(field_name) != expected_value:
                        raise ClaimValidationError(
                            f"incomplete output owner marker mismatch: {field_name}"
                        )
                marker_owner_uuid = _validate_uuid(
                    marker.get("owner_uuid"), "incomplete marker owner UUID"
                )
                marker_attempt_uuid = _validate_uuid(
                    marker.get("attempt_uuid"),
                    "incomplete marker submission attempt UUID",
                )
                marker_matches_pending_owner = marker_owner_uuid == exact_owner_uuid
                if marker_matches_pending_owner:
                    if marker_attempt_uuid != current_attempt_uuid:
                        raise ClaimValidationError(
                            "current owner marker has the wrong submission attempt"
                        )
                else:
                    previous_attempts = [
                        item
                        for item in attempts
                        if item.get("attempt_uuid") == marker_attempt_uuid
                    ]
                    try:
                        previous_attempt_number = int(
                            previous_attempts[0].get("attempt_number", 0)
                        ) if len(previous_attempts) == 1 else 0
                        current_attempt_number = int(
                            current_attempts[0].get("attempt_number", 0)
                        )
                    except (TypeError, ValueError) as exc:
                        raise ClaimValidationError(
                            "submission attempt number is malformed"
                        ) from exc
                    previous_resume_marker = bool(
                        partial_acquisition
                        and submission_kind == "resume"
                        and len(previous_attempts) == 1
                        and marker_attempt_uuid != current_attempt_uuid
                        and previous_attempt_number < current_attempt_number
                    )
                    if not previous_resume_marker:
                        raise ClaimValidationError(
                            "incomplete marker belongs to an unexpected output owner"
                        )
        if not partial_acquisition and not (incomplete_present and marker_present):
            raise ClaimBlockedError(
                "established output owner lacks its resumable .incomplete marker"
            )
        orphan_lease = _acquire_output_owner_lease(plan.claim_dir)
        try:
            target = (
                ClaimState.INTERRUPTED
                if current in {ClaimState.SUBMITTED, ClaimState.RUNNING}
                else current
            )
            state = _transition_locked(
                plan.claim_dir,
                claim_uuid,
                target,
                event_type="ORPHANED_OUTPUT_OWNER_REVOCATION_STARTED",
                updates={
                    "pending_owner_uuid": exact_owner_uuid,
                    "job_failure_reason": machine_reason,
                    "owner_revocation_scheduler_evidence": scheduler.to_dict(),
                },
                details={
                    "owner_uuid": exact_owner_uuid,
                    "reason": machine_reason,
                    "scheduler_evidence": scheduler.to_dict(),
                    "owner_lease_verified_released": True,
                    "partial_owner_acquisition": partial_acquisition,
                    "incomplete_present": incomplete_present,
                    "owner_marker_present": marker_present,
                    "marker_matches_pending_owner": marker_matches_pending_owner,
                    "submission_kind": submission_kind,
                },
                allow_same=target is current,
            )
            _archive_active_directory(
                plan.claim_dir, "active_owner", exact_owner_uuid
            )
            state = _transition_locked(
                plan.claim_dir,
                claim_uuid,
                target,
                event_type="ORPHANED_OUTPUT_OWNER_REVOKED",
                updates={
                    "active_owner_uuid": None,
                    "pending_owner_uuid": None,
                    "job_failure_reason": machine_reason,
                    "owner_revocation_scheduler_evidence": scheduler.to_dict(),
                },
                details={
                    "owner_uuid": exact_owner_uuid,
                    "reason": machine_reason,
                    "scheduler_evidence": scheduler.to_dict(),
                    "owner_lease_verified_released": True,
                    "active_owner_archived": True,
                    "partial_owner_acquisition": partial_acquisition,
                },
                allow_same=True,
            )
            return state
        finally:
            _release_owner_lease(orphan_lease)


def _resume_checks(
    plan: ClaimPlan,
    *,
    claim_uuid: str,
    scheduler: SchedulerObservation,
) -> tuple[dict[str, Any], dict[str, Any], int, str]:
    _validate_plan_geometry(plan)
    identity, state = _load_claim(plan.claim_dir)
    if not _scheduler_absence_is_current(scheduler, identity, state):
        raise ClaimBlockedError(
            "resume requires fresh scheduler ABSENT evidence bound to the current claim revision"
        )
    if identity.get("claim_uuid") != claim_uuid:
        raise ClaimValidationError("resume claim UUID mismatch")
    if identity.get("canonical_run_id") != plan.run_id:
        raise ClaimValidationError("resume run ID mismatch")
    exact_bindings = {
        "canonical_thermal_run_root": str(plan.run_root),
        "canonical_expected_output_path": str(plan.expected_output_path),
        "canonical_incomplete_output_path": str(plan.incomplete_output_path),
        "base_resolved_spec_sha256": plan.base_resolved_spec_sha256,
        "execution_resolved_spec_sha256": plan.execution_resolved_spec_sha256,
        "frozen_config_sha256": plan.frozen_config_sha256,
        "thermal_implementation_inventory_sha256": (
            plan.thermal_implementation_inventory_sha256
        ),
        "job_script_sha256": plan.job_script_sha256,
        "run_class": plan.run_class,
        "scientific_eligible": plan.scientific_eligible,
        "reuse_eligible": plan.reuse_eligible,
        "scheduler_job_name": plan.scheduler_job_name,
    }
    for field, expected in exact_bindings.items():
        if identity.get(field) != expected:
            raise ClaimConflictError(f"resume authorization changed: {field}")
    current = ClaimState(str(state["state"]))
    if current not in {
        ClaimState.SUBMISSION_FAILED,
        ClaimState.FAILED,
        ClaimState.INTERRUPTED,
        ClaimState.RECONCILED,
    }:
        raise ClaimBlockedError(f"claim state {current.value} is not resumable")
    _reject_symlink(plan.expected_output_path, "final output")
    _reject_symlink(plan.incomplete_output_path, "incomplete output")
    if _lexists(plan.expected_output_path):
        raise ClaimBlockedError("a finalized run is not resumable")
    _read_resumable_output_marker(plan, identity)
    if _active_owner_record(plan.claim_dir) is not None:
        raise ClaimBlockedError("an active output owner blocks resume")
    if _lexists(_active_reservation_path(plan.claim_dir)):
        raise ClaimBlockedError("an active submission reservation blocks resume")
    next_attempt = int(state.get("submission_attempt_number", 0)) + 1
    next_uuid = _attempt_uuid(claim_uuid, next_attempt)
    return identity, state, next_attempt, next_uuid


def plan_resume_submission(
    plan: ClaimPlan,
    *,
    claim_uuid: str,
    scheduler: SchedulerObservation,
) -> dict[str, Any]:
    """Read-only explicit resume authorization planning."""

    _identity, _state, number, attempt_uuid = _resume_checks(
        plan, claim_uuid=claim_uuid, scheduler=scheduler
    )
    return {
        "allowed": True,
        "claim_path": str(plan.claim_dir),
        "claim_uuid": claim_uuid,
        "run_id": plan.run_id,
        "execution_resolved_spec_sha256": plan.execution_resolved_spec_sha256,
        "next_submission_attempt_number": number,
        "next_submission_attempt_uuid": attempt_uuid,
        "scheduler_evidence": scheduler.to_dict(),
    }


def acquire_resume_submission(
    plan: ClaimPlan,
    *,
    claim_uuid: str,
    scheduler: SchedulerObservation,
) -> ClaimHandle:
    if not plan.complete_for_acquisition:
        raise ClaimValidationError("resume acquisition requires all exact hashes")
    reservation_token = secrets.token_hex(32)
    with _transition_lock(plan.claim_dir):
        _identity, state, number, attempt_uuid = _resume_checks(
            plan, claim_uuid=claim_uuid, scheduler=scheduler
        )
        _create_active_reservation(
            plan.claim_dir,
            claim_uuid=claim_uuid,
            attempt_number=number,
            attempt_uuid=attempt_uuid,
            submission_kind="resume",
            reservation_token=reservation_token,
        )
        attempts = list(state.get("attempts", []))
        attempts.append(
            {
                "attempt_number": number,
                "attempt_uuid": attempt_uuid,
                "submission_kind": "resume",
                "reserved_at_utc": _utc_now(),
                "qsub_invoked": False,
                "scheduler_job_id": None,
                "scheduler_precheck": scheduler.to_dict(),
                "result": "RESERVED",
            }
        )
        _transition_locked(
            plan.claim_dir,
            claim_uuid,
            ClaimState.RESERVED,
            event_type="RESUME_SUBMISSION_RESERVED",
            updates={
                "submission_attempt_number": number,
                "current_attempt_uuid": attempt_uuid,
                "scheduler_job_id": None,
                "pending_owner_uuid": None,
                "attempts": attempts,
            },
            details={
                "attempt_uuid": attempt_uuid,
                "scheduler_evidence": scheduler.to_dict(),
            },
        )
    return ClaimHandle(
        plan,
        number,
        attempt_uuid,
        "resume",
        reservation_token,
    )


def _submission_outcome_is_pending(state: Mapping[str, Any]) -> bool:
    if ClaimState(str(state.get("state"))) is not ClaimState.RESERVED:
        return False
    current_uuid = state.get("current_attempt_uuid")
    attempts = state.get("attempts")
    if not isinstance(attempts, list) or any(
        not isinstance(item, dict) for item in attempts
    ):
        raise ClaimValidationError("claim attempts are malformed")
    matches = [item for item in attempts if item.get("attempt_uuid") == current_uuid]
    if len(matches) != 1:
        raise ClaimValidationError("current claim attempt is missing or duplicated")
    invoked = matches[0].get("qsub_invoked")
    result = matches[0].get("result")
    if invoked is True:
        if result != "INVOKED_OUTCOME_PENDING":
            raise ClaimValidationError("pending submission outcome is inconsistent")
        return True
    if invoked is not False or result != "RESERVED":
        raise ClaimValidationError("reserved submission attempt is inconsistent")
    return False


def _administratively_mark_pending_unknown(
    claim_dir: Path,
    claim_uuid: str,
    state: Mapping[str, Any],
    *,
    administrative_override: bool,
    operation: str,
) -> dict[str, Any]:
    if not _submission_outcome_is_pending(state):
        return dict(state)
    if not administrative_override:
        raise ClaimBlockedError(
            "an invoked qsub attempt still has a pending outcome"
        )
    return _transition_locked(
        claim_dir,
        claim_uuid,
        ClaimState.SUBMISSION_UNKNOWN,
        event_type="ADMINISTRATIVE_PENDING_SUBMISSION_MARKED_UNKNOWN",
        updates={"pending_submission_override_operation": operation},
        details={
            "administrative_override": True,
            "operation": operation,
            "orphan_scheduler_job_risk": True,
        },
    )


def reconcile_claim(
    plan: ClaimPlan,
    *,
    claim_uuid: str,
    scheduler: SchedulerObservation,
    confirmation: str,
    reason: str,
    administrative_override: bool = False,
) -> dict[str, Any]:
    """Conservatively reconcile a nonterminal claim; never deletes history."""

    _validate_plan_geometry(plan)
    if confirmation != (
        "ADMIN_OVERRIDE_RECONCILE" if administrative_override else "RECONCILE"
    ):
        raise ClaimValidationError("reconciliation confirmation token is invalid")
    machine_reason = _validate_reason(reason)
    with _transition_lock(plan.claim_dir):
        identity, state = _load_claim(plan.claim_dir)
        if identity.get("claim_uuid") != claim_uuid:
            raise ClaimValidationError("reconciliation claim UUID mismatch")
        current = ClaimState(str(state["state"]))
        if current in {ClaimState.COMPLETE, ClaimState.RELEASED}:
            raise ClaimBlockedError(f"terminal claim {current.value} cannot be reconciled")
        if not administrative_override and not _scheduler_absence_is_current(
            scheduler, identity, state
        ):
            raise ClaimBlockedError(
                "reconciliation requires scheduler ABSENT evidence bound to the current claim revision"
            )
        state = _administratively_mark_pending_unknown(
            plan.claim_dir,
            claim_uuid,
            state,
            administrative_override=administrative_override,
            operation="reconcile",
        )
        if _lexists(plan.expected_output_path):
            raise ClaimBlockedError("a finalized run cannot be reconciled")
        if _active_owner_record(plan.claim_dir) is not None:
            raise ClaimBlockedError("active output owner blocks reconciliation")
        _archive_current_submission_reservation(
            plan.claim_dir, claim_uuid, state
        )
        result = _transition_locked(
            plan.claim_dir,
            claim_uuid,
            ClaimState.RECONCILED,
            event_type="ADMINISTRATIVE_RECONCILIATION"
            if administrative_override
            else "OWNER_SAFE_RECONCILIATION",
            updates={
                "reconciliation_reason": machine_reason,
                "reconciliation_scheduler_evidence": scheduler.to_dict(),
                "administrative_override": administrative_override,
            },
            details={
                "reason": machine_reason,
                "scheduler_evidence": scheduler.to_dict(),
                "administrative_override": administrative_override,
            },
        )
    return result


def release_claim(
    plan: ClaimPlan,
    *,
    claim_uuid: str,
    scheduler: SchedulerObservation,
    confirmation: str,
    reason: str,
    administrative_override: bool = False,
) -> dict[str, Any]:
    """Owner-safe terminal release; the claim directory remains a tombstone."""

    _validate_plan_geometry(plan)
    expected_confirmation = (
        "ADMIN_OVERRIDE_RELEASE" if administrative_override else "RELEASE"
    )
    if confirmation != expected_confirmation:
        raise ClaimValidationError("release confirmation token is invalid")
    machine_reason = _validate_reason(reason)
    with _transition_lock(plan.claim_dir):
        identity, state = _load_claim(plan.claim_dir)
        if identity.get("claim_uuid") != claim_uuid:
            raise ClaimValidationError("release claim UUID mismatch")
        current = ClaimState(str(state["state"]))
        if current in {ClaimState.COMPLETE, ClaimState.RELEASED}:
            raise ClaimBlockedError(f"terminal claim {current.value} cannot be released")
        if not administrative_override and not _scheduler_absence_is_current(
            scheduler, identity, state
        ):
            raise ClaimBlockedError(
                "release requires scheduler ABSENT evidence bound to the current claim revision"
            )
        state = _administratively_mark_pending_unknown(
            plan.claim_dir,
            claim_uuid,
            state,
            administrative_override=administrative_override,
            operation="release",
        )
        if _lexists(plan.expected_output_path):
            raise ClaimBlockedError("a finalized run cannot be released")
        if _active_owner_record(plan.claim_dir) is not None:
            raise ClaimBlockedError("active output owner blocks release")
        _archive_current_submission_reservation(
            plan.claim_dir, claim_uuid, state
        )
        result = _transition_locked(
            plan.claim_dir,
            claim_uuid,
            ClaimState.RELEASED,
            event_type="ADMINISTRATIVE_OVERRIDE_RELEASE"
            if administrative_override
            else "OWNER_SAFE_RELEASE",
            updates={
                "release_reason": machine_reason,
                "release_scheduler_evidence": scheduler.to_dict(),
                "administrative_override": administrative_override,
            },
            details={
                "reason": machine_reason,
                "scheduler_evidence": scheduler.to_dict(),
                "administrative_override": administrative_override,
            },
        )
    return result
