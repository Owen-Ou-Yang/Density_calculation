"""Hash-based provenance records for thermal calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


class ProvenanceError(ValueError):
    """Raised when provenance cannot be validated or verified."""


class ExecutionStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QCStatus(str, Enum):
    NOT_EVALUATED = "not_evaluated"
    PASSED = "passed"
    FAILED = "failed"


class AnalysisStatus(str, Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def normalize_sha256(value: object, field_name: str = "sha256") -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ProvenanceError(f"{field_name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 of a regular file without loading it all into memory."""

    file_path = Path(path)
    if not file_path.is_file():
        raise ProvenanceError(f"artifact is not a regular file: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_ready(value.to_dict())
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ProvenanceError(f"value is not canonically JSON serializable: {type(value).__name__}")


def canonical_sha256(value: Any) -> str:
    """Hash deterministic, whitespace-free JSON for a configuration or plan."""

    encoded = json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ArtifactProvenance:
    name: str
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ProvenanceError("artifact name must be a nonempty string")
        if not isinstance(self.path, str) or not self.path.strip():
            raise ProvenanceError("artifact path must be a nonempty string")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "path", self.path.strip())
        object.__setattr__(self, "sha256", normalize_sha256(self.sha256))
        if isinstance(self.size_bytes, bool) or int(self.size_bytes) < 0:
            raise ProvenanceError("artifact size_bytes must be a nonnegative integer")
        object.__setattr__(self, "size_bytes", int(self.size_bytes))

    @classmethod
    def from_file(
        cls,
        name: str,
        path: str | Path,
        *,
        relative_to: str | Path | None = None,
    ) -> "ArtifactProvenance":
        file_path = Path(path)
        if not file_path.is_file():
            raise ProvenanceError(f"artifact is not a regular file: {file_path}")
        stored_path = file_path
        if relative_to is not None:
            try:
                stored_path = file_path.resolve().relative_to(Path(relative_to).resolve())
            except ValueError as exc:
                raise ProvenanceError(
                    f"artifact {file_path} is outside relative_to={relative_to}"
                ) from exc
        return cls(
            name=name,
            path=str(stored_path),
            sha256=sha256_file(file_path),
            size_bytes=file_path.stat().st_size,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ArtifactProvenance":
        try:
            return cls(
                name=mapping["name"],
                path=mapping["path"],
                sha256=mapping["sha256"],
                size_bytes=mapping["size_bytes"],
            )
        except KeyError as exc:
            raise ProvenanceError(f"artifact provenance missing field: {exc.args[0]}") from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class RunProvenance:
    """Identity, gate states, and artifacts for one replica run."""

    run_id: str
    replica_id: str
    protocol_sha256: str
    input_snapshot_sha256: str
    model_sha256: str
    execution_status: ExecutionStatus = ExecutionStatus.PLANNED
    qc_status: QCStatus = QCStatus.NOT_EVALUATED
    analysis_status: AnalysisStatus = AnalysisStatus.NOT_STARTED
    artifacts: tuple[ArtifactProvenance, ...] = field(default_factory=tuple)
    created_utc: str = field(default_factory=_utc_now)
    schema_version: int = 1

    def __post_init__(self) -> None:
        for field_name in ("run_id", "replica_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ProvenanceError(f"{field_name} must be a nonempty string")
            object.__setattr__(self, field_name, value.strip())
        for field_name in (
            "protocol_sha256",
            "input_snapshot_sha256",
            "model_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                normalize_sha256(getattr(self, field_name), field_name),
            )
        try:
            object.__setattr__(self, "execution_status", ExecutionStatus(self.execution_status))
            object.__setattr__(self, "qc_status", QCStatus(self.qc_status))
            object.__setattr__(self, "analysis_status", AnalysisStatus(self.analysis_status))
        except ValueError as exc:
            raise ProvenanceError(f"invalid run status: {exc}") from exc

        artifacts = tuple(self.artifacts)
        if not all(isinstance(artifact, ArtifactProvenance) for artifact in artifacts):
            raise ProvenanceError("artifacts must contain only ArtifactProvenance records")
        artifact_names = [artifact.name for artifact in artifacts]
        if len(set(artifact_names)) != len(artifact_names):
            raise ProvenanceError("artifact names must be unique within a run")
        object.__setattr__(self, "artifacts", artifacts)

        if self.schema_version != 1:
            raise ProvenanceError(
                f"unsupported provenance schema_version={self.schema_version}"
            )
        try:
            datetime.fromisoformat(self.created_utc.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ProvenanceError("created_utc must be an ISO-8601 timestamp") from exc

    def artifact_named(self, name: str) -> ArtifactProvenance | None:
        for artifact in self.artifacts:
            if artifact.name == name:
                return artifact
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "replica_id": self.replica_id,
            "protocol_sha256": self.protocol_sha256,
            "input_snapshot_sha256": self.input_snapshot_sha256,
            "model_sha256": self.model_sha256,
            "execution_status": self.execution_status.value,
            "qc_status": self.qc_status.value,
            "analysis_status": self.analysis_status.value,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "created_utc": self.created_utc,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RunProvenance":
        required = {
            "run_id",
            "replica_id",
            "protocol_sha256",
            "input_snapshot_sha256",
            "model_sha256",
            "execution_status",
            "qc_status",
            "analysis_status",
        }
        missing = sorted(required.difference(mapping))
        if missing:
            raise ProvenanceError(f"run provenance missing required fields: {missing}")
        raw_artifacts = mapping.get("artifacts", [])
        if not isinstance(raw_artifacts, list):
            raise ProvenanceError("artifacts must be a list")
        return cls(
            run_id=mapping["run_id"],
            replica_id=mapping["replica_id"],
            protocol_sha256=mapping["protocol_sha256"],
            input_snapshot_sha256=mapping["input_snapshot_sha256"],
            model_sha256=mapping["model_sha256"],
            execution_status=mapping["execution_status"],
            qc_status=mapping["qc_status"],
            analysis_status=mapping["analysis_status"],
            artifacts=tuple(ArtifactProvenance.from_mapping(item) for item in raw_artifacts),
            created_utc=mapping.get("created_utc", _utc_now()),
            schema_version=int(mapping.get("schema_version", 1)),
        )


def write_provenance(path: str | Path, record: RunProvenance) -> None:
    """Atomically write a provenance JSON document."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    payload = json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n"
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_provenance(path: str | Path) -> RunProvenance:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"failed to read provenance: {source}") from exc
    if not isinstance(payload, Mapping):
        raise ProvenanceError("provenance JSON root must be an object")
    return RunProvenance.from_mapping(payload)


@dataclass(frozen=True)
class StatePointProvenance:
    """Finalized identity and gates for one reusable NPT production plateau.

    This is deliberately state-point scoped.  A Tg sweep and a density-only
    run have different sweep hashes, but their individual density plateaus can
    still be evaluated against an explicitly requested state-point identity.
    """

    run_id: str
    replica_id: str
    state_point_id: str
    trajectory_key: str
    density_reuse_key: str
    density_qc_policy_id: str
    density_qc_policy_sha256: str
    density_qc_implementation_sha256: str
    history_key: str
    run_manifest_relpath: str
    input_snapshot_sha256: str
    model_sha256: str
    run_class: str
    scientific_eligible: bool
    reuse_eligible: bool
    execution_status: ExecutionStatus
    qc_status: QCStatus
    analysis_status: AnalysisStatus = AnalysisStatus.NOT_APPLICABLE
    artifacts: tuple[ArtifactProvenance, ...] = field(default_factory=tuple)
    created_utc: str = field(default_factory=_utc_now)
    schema_version: str = "thermal-properties-state-point-provenance/v3"

    def __post_init__(self) -> None:
        for field_name in ("run_id", "replica_id", "state_point_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ProvenanceError(f"{field_name} must be a nonempty string")
            object.__setattr__(self, field_name, value.strip())
        allowed_classes = {
            "ENGINEERING_SMOKE",
            "DEBUG",
            "PILOT",
            "PRODUCTION",
        }
        if self.run_class not in allowed_classes:
            raise ProvenanceError(f"unsupported run_class: {self.run_class}")
        if type(self.scientific_eligible) is not bool or type(
            self.reuse_eligible
        ) is not bool:
            raise ProvenanceError(
                "scientific_eligible and reuse_eligible must be explicit booleans"
            )
        if self.run_class != "PRODUCTION" and (
            self.scientific_eligible or self.reuse_eligible
        ):
            raise ProvenanceError(
                f"{self.run_class} provenance cannot be scientific or reusable"
            )
        if self.run_class == "PRODUCTION" and not self.scientific_eligible:
            raise ProvenanceError(
                "PRODUCTION provenance must be scientifically eligible"
            )
        if self.reuse_eligible and not self.scientific_eligible:
            raise ProvenanceError(
                "reusable provenance must be scientifically eligible"
            )
        if (
            not isinstance(self.density_qc_policy_id, str)
            or not self.density_qc_policy_id.strip()
        ):
            raise ProvenanceError(
                "density_qc_policy_id must be a nonempty string"
            )
        object.__setattr__(
            self, "density_qc_policy_id", self.density_qc_policy_id.strip()
        )
        if (
            not isinstance(self.run_manifest_relpath, str)
            or not self.run_manifest_relpath.strip()
            or Path(self.run_manifest_relpath).is_absolute()
        ):
            raise ProvenanceError(
                "run_manifest_relpath must be a nonempty relative path"
            )
        for field_name in (
            "trajectory_key",
            "density_reuse_key",
            "density_qc_policy_sha256",
            "density_qc_implementation_sha256",
            "history_key",
            "input_snapshot_sha256",
            "model_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                normalize_sha256(getattr(self, field_name), field_name),
            )
        try:
            object.__setattr__(self, "execution_status", ExecutionStatus(self.execution_status))
            object.__setattr__(self, "qc_status", QCStatus(self.qc_status))
            object.__setattr__(self, "analysis_status", AnalysisStatus(self.analysis_status))
        except ValueError as exc:
            raise ProvenanceError(f"invalid state-point status: {exc}") from exc
        artifacts = tuple(self.artifacts)
        if not artifacts or not all(
            isinstance(artifact, ArtifactProvenance) for artifact in artifacts
        ):
            raise ProvenanceError(
                "state-point artifacts must contain ArtifactProvenance records"
            )
        names = [artifact.name for artifact in artifacts]
        if len(set(names)) != len(names):
            raise ProvenanceError("state-point artifact names must be unique")
        object.__setattr__(self, "artifacts", artifacts)
        if self.schema_version != "thermal-properties-state-point-provenance/v3":
            raise ProvenanceError(
                f"unsupported state-point provenance schema: {self.schema_version}"
            )

    def artifact_named(self, name: str) -> ArtifactProvenance | None:
        return next((item for item in self.artifacts if item.name == name), None)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "replica_id": self.replica_id,
            "state_point_id": self.state_point_id,
            "trajectory_key": self.trajectory_key,
            "density_reuse_key": self.density_reuse_key,
            "density_qc_policy_id": self.density_qc_policy_id,
            "density_qc_policy_sha256": self.density_qc_policy_sha256,
            "density_qc_implementation_sha256": (
                self.density_qc_implementation_sha256
            ),
            "history_key": self.history_key,
            "run_manifest_relpath": self.run_manifest_relpath,
            "input_snapshot_sha256": self.input_snapshot_sha256,
            "model_sha256": self.model_sha256,
            "run_class": self.run_class,
            "scientific_eligible": self.scientific_eligible,
            "reuse_eligible": self.reuse_eligible,
            "execution_status": self.execution_status.value,
            "qc_status": self.qc_status.value,
            "analysis_status": self.analysis_status.value,
            "artifacts": [item.to_dict() for item in self.artifacts],
            "created_utc": self.created_utc,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "StatePointProvenance":
        required = {
            "run_id",
            "replica_id",
            "state_point_id",
            "trajectory_key",
            "density_reuse_key",
            "density_qc_policy_id",
            "density_qc_policy_sha256",
            "density_qc_implementation_sha256",
            "history_key",
            "run_manifest_relpath",
            "input_snapshot_sha256",
            "model_sha256",
            "run_class",
            "scientific_eligible",
            "reuse_eligible",
            "execution_status",
            "qc_status",
            "analysis_status",
            "artifacts",
        }
        missing = sorted(required.difference(mapping))
        if missing:
            raise ProvenanceError(
                f"state-point provenance missing fields: {missing}"
            )
        artifacts = mapping["artifacts"]
        if not isinstance(artifacts, list):
            raise ProvenanceError("state-point artifacts must be an array")
        return cls(
            run_id=mapping["run_id"],
            replica_id=mapping["replica_id"],
            state_point_id=mapping["state_point_id"],
            trajectory_key=mapping["trajectory_key"],
            density_reuse_key=mapping["density_reuse_key"],
            density_qc_policy_id=mapping["density_qc_policy_id"],
            density_qc_policy_sha256=mapping["density_qc_policy_sha256"],
            density_qc_implementation_sha256=mapping[
                "density_qc_implementation_sha256"
            ],
            history_key=mapping["history_key"],
            run_manifest_relpath=mapping["run_manifest_relpath"],
            input_snapshot_sha256=mapping["input_snapshot_sha256"],
            model_sha256=mapping["model_sha256"],
            run_class=mapping["run_class"],
            scientific_eligible=mapping["scientific_eligible"],
            reuse_eligible=mapping["reuse_eligible"],
            execution_status=mapping["execution_status"],
            qc_status=mapping["qc_status"],
            analysis_status=mapping["analysis_status"],
            artifacts=tuple(ArtifactProvenance.from_mapping(item) for item in artifacts),
            created_utc=mapping.get("created_utc", _utc_now()),
            schema_version=mapping.get(
                "schema_version", "thermal-properties-state-point-provenance/v3"
            ),
        )


def write_state_point_provenance(
    path: str | Path, record: StatePointProvenance
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_state_point_provenance(path: str | Path) -> StatePointProvenance:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(
            f"failed to read state-point provenance: {source}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ProvenanceError("state-point provenance root must be an object")
    return StatePointProvenance.from_mapping(payload)
