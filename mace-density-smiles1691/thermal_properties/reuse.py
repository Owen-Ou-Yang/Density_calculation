"""Conservative, noninteractive cross-run artifact reuse."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .provenance import (
    AnalysisStatus,
    ArtifactProvenance,
    ExecutionStatus,
    ProvenanceError,
    QCStatus,
    StatePointProvenance,
    canonical_sha256,
    normalize_sha256,
    read_state_point_provenance,
    sha256_file,
)


PROVENANCE_FILENAME = "state_point_provenance.json"


def density_qc_implementation_sha256(
    execution_identity: Mapping[str, Any],
) -> str:
    """Hash the exact code sources that implement state-point density QC."""

    orchestration = execution_identity.get("orchestration_source_sha256")
    quality_control = execution_identity.get("quality_control_source_sha256")
    if not isinstance(orchestration, Mapping) or not isinstance(
        quality_control, Mapping
    ):
        raise ReuseError("execution identity lacks density-QC source hashes")
    simulation_sha256 = orchestration.get("simulation.py")
    try:
        normalized_simulation = normalize_sha256(
            simulation_sha256, "orchestration_source_sha256.simulation.py"
        )
        normalized_qc_sources = {
            str(name): normalize_sha256(value, f"quality_control_source.{name}")
            for name, value in quality_control.items()
        }
    except ProvenanceError as exc:
        raise ReuseError(str(exc)) from exc
    if not normalized_qc_sources:
        raise ReuseError("execution identity has no density-QC source hashes")
    return canonical_sha256(
        {
            "schema_version": "thermal-properties-density-qc-implementation/v1",
            "simulation.py": normalized_simulation,
            "quality_control_source_sha256": normalized_qc_sources,
        }
    )


class ReuseError(RuntimeError):
    """Base class for reuse resolution failures."""


class AmbiguousReuseError(ReuseError):
    """Raised when more than one exact reusable candidate exists."""


class ReuseAction(str, Enum):
    RUN_NEW = "run_new"
    REUSE = "reuse"


@dataclass(frozen=True)
class ReuseRequest:
    replica_id: str
    run_class: str
    scientific_eligible: bool
    reuse_eligible: bool
    density_reuse_key: str
    density_qc_policy_id: str
    density_qc_policy_sha256: str
    density_qc_implementation_sha256: str
    input_snapshot_sha256: str
    model_sha256: str
    artifact_name: str
    require_analysis_success: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.replica_id, str) or not self.replica_id.strip():
            raise ReuseError("replica_id must be a nonempty string")
        if not isinstance(self.artifact_name, str) or not self.artifact_name.strip():
            raise ReuseError("artifact_name must be a nonempty string")
        if (
            not isinstance(self.density_qc_policy_id, str)
            or not self.density_qc_policy_id.strip()
        ):
            raise ReuseError("density_qc_policy_id must be a nonempty string")
        object.__setattr__(self, "replica_id", self.replica_id.strip())
        object.__setattr__(self, "artifact_name", self.artifact_name.strip())
        object.__setattr__(
            self, "density_qc_policy_id", self.density_qc_policy_id.strip()
        )
        if self.run_class != "PRODUCTION":
            raise ReuseError(
                "automatic cross-run reuse requests require run_class=PRODUCTION"
            )
        if self.scientific_eligible is not True or self.reuse_eligible is not True:
            raise ReuseError(
                "automatic cross-run reuse requests require both eligibility gates"
            )
        for field_name in (
            "density_reuse_key",
            "density_qc_policy_sha256",
            "density_qc_implementation_sha256",
            "input_snapshot_sha256",
            "model_sha256",
        ):
            try:
                normalized = normalize_sha256(getattr(self, field_name), field_name)
            except ProvenanceError as exc:
                raise ReuseError(str(exc)) from exc
            object.__setattr__(self, field_name, normalized)
        if type(self.require_analysis_success) is not bool:
            raise ReuseError("require_analysis_success must be an explicit boolean")


@dataclass(frozen=True)
class ReuseCandidate:
    provenance_path: Path
    provenance: StatePointProvenance
    artifact: ArtifactProvenance

    @property
    def artifact_path(self) -> Path:
        path = Path(self.artifact.path)
        if path.is_absolute():
            return path
        return self.provenance_path.parent / path

    @property
    def run_manifest_path(self) -> Path:
        return (
            self.provenance_path.parent / self.provenance.run_manifest_relpath
        ).resolve()


@dataclass(frozen=True)
class ReuseDecision:
    action: ReuseAction
    candidate: ReuseCandidate | None
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", ReuseAction(self.action))
        if self.action is ReuseAction.RUN_NEW and self.candidate is not None:
            raise ReuseError("RUN_NEW decision cannot contain a candidate")
        if self.action is ReuseAction.REUSE and self.candidate is None:
            raise ReuseError("REUSE decision requires a candidate")


def _identity_matches(record: StatePointProvenance, request: ReuseRequest) -> bool:
    return (
        record.replica_id == request.replica_id
        and record.run_class == request.run_class
        and record.scientific_eligible is request.scientific_eligible
        and record.reuse_eligible is request.reuse_eligible
        and record.density_reuse_key == request.density_reuse_key
        and record.density_qc_policy_id == request.density_qc_policy_id
        and record.density_qc_policy_sha256
        == request.density_qc_policy_sha256
        and record.density_qc_implementation_sha256
        == request.density_qc_implementation_sha256
        and record.input_snapshot_sha256 == request.input_snapshot_sha256
        and record.model_sha256 == request.model_sha256
    )


def _eligible_candidate(
    provenance_path: Path,
    record: StatePointProvenance,
    request: ReuseRequest,
) -> ReuseCandidate | None:
    if not _identity_matches(record, request):
        return None
    if record.execution_status is not ExecutionStatus.SUCCEEDED:
        return None
    if record.qc_status is not QCStatus.PASSED:
        return None
    if (
        request.require_analysis_success
        and record.analysis_status is not AnalysisStatus.SUCCEEDED
    ):
        return None
    artifact = record.artifact_named(request.artifact_name)
    if artifact is None:
        return None

    candidate = ReuseCandidate(
        provenance_path=provenance_path,
        provenance=record,
        artifact=artifact,
    )
    if any(part.endswith(".incomplete") for part in candidate.provenance_path.parts):
        return None
    artifact_path = candidate.artifact_path
    if not artifact_path.is_file():
        return None
    if artifact_path.stat().st_size != artifact.size_bytes:
        return None
    try:
        actual_sha256 = sha256_file(artifact_path)
    except ProvenanceError:
        return None
    if actual_sha256 != artifact.sha256:
        return None
    density_qc_artifact = record.artifact_named("density_qc_result")
    if density_qc_artifact is None:
        return None
    density_qc_path = Path(density_qc_artifact.path)
    if not density_qc_path.is_absolute():
        density_qc_path = provenance_path.parent / density_qc_path
    if (
        not density_qc_path.is_file()
        or density_qc_path.stat().st_size != density_qc_artifact.size_bytes
        or sha256_file(density_qc_path) != density_qc_artifact.sha256
    ):
        return None
    try:
        density_qc_payload = json.loads(
            density_qc_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    raw_policy_results = (
        density_qc_payload.get("policy_results")
        if isinstance(density_qc_payload, dict)
        else None
    )
    matching_density_policies = [
        item
        for item in raw_policy_results
        if isinstance(item, dict)
        and item.get("role") == "density_target"
        and item.get("policy_id") == request.density_qc_policy_id
        and item.get("policy_sha256") == request.density_qc_policy_sha256
        and str(item.get("status", "")).upper() == "PASS"
    ] if isinstance(raw_policy_results, list) else []
    if (
        str(density_qc_payload.get("status", "")).upper() != "PASS"
        or density_qc_payload.get("qc_implementation_sha256")
        != request.density_qc_implementation_sha256
        or len(matching_density_policies) != 1
    ):
        return None
    if not candidate.run_manifest_path.is_file():
        return None
    digest_path = candidate.run_manifest_path.with_name("run_manifest.sha256")
    if not digest_path.is_file():
        return None
    try:
        declared_manifest_sha256 = normalize_sha256(
            digest_path.read_text(encoding="ascii").strip(),
            "run_manifest.sha256",
        )
        if sha256_file(candidate.run_manifest_path) != declared_manifest_sha256:
            return None
    except (OSError, UnicodeError, ProvenanceError):
        return None
    try:
        run_manifest = json.loads(
            candidate.run_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(run_manifest, dict) or run_manifest.get("finalized") is not True:
        return None
    if (
        run_manifest.get("run_class") != record.run_class
        or run_manifest.get("scientific_eligible")
        is not record.scientific_eligible
        or run_manifest.get("reuse_eligible") is not record.reuse_eligible
        or run_manifest.get("run_class") != "PRODUCTION"
        or run_manifest.get("scientific_eligible") is not True
        or run_manifest.get("reuse_eligible") is not True
    ):
        return None
    execution_identity = run_manifest.get("execution_identity")
    if not isinstance(execution_identity, Mapping):
        return None
    try:
        manifested_qc_implementation = density_qc_implementation_sha256(
            execution_identity
        )
    except ReuseError:
        return None
    if (
        manifested_qc_implementation
        != request.density_qc_implementation_sha256
        or manifested_qc_implementation
        != record.density_qc_implementation_sha256
    ):
        return None
    if str(run_manifest.get("status", "")).upper() not in {
        "COMPLETE",
        "COMPLETE_WITH_QC_FAILURE",
    }:
        return None
    run_root = candidate.run_manifest_path.parent.parent.resolve()
    try:
        candidate.provenance_path.resolve().relative_to(run_root)
    except ValueError:
        return None
    if run_manifest.get("run_id") != record.run_id:
        return None
    state_points = run_manifest.get("state_points")
    if not isinstance(state_points, list):
        return None
    state_matches = [
        item
        for item in state_points
        if isinstance(item, dict)
        and item.get("state_point_id") == record.state_point_id
    ]
    if len(state_matches) != 1:
        return None
    state = state_matches[0]
    roles = state.get("roles", [])
    policy_results = state.get("qc_policy_results")
    density_policy_results = [
        item
        for item in policy_results
        if isinstance(item, dict) and item.get("role") == "density_target"
    ] if isinstance(policy_results, list) else []
    if (
        not isinstance(roles, list)
        or "density_target" not in [str(role).lower() for role in roles]
        or state.get("use_for_density") is not True
        or state.get("status") != "COMPLETE"
        or state.get("validated") is not True
        or state.get("replica_id") != record.replica_id
        or state.get("run_class") != record.run_class
        or state.get("scientific_eligible") is not True
        or state.get("reuse_eligible") is not True
        or str(state.get("qc_status", "")).upper() != "PASS"
        or state.get("density_reuse_key") != request.density_reuse_key
        or state.get("trajectory_key") != record.trajectory_key
        or len(density_policy_results) != 1
        or density_policy_results[0].get("status") != "PASS"
        or density_policy_results[0].get("policy_id")
        != request.density_qc_policy_id
        or density_policy_results[0].get("policy_sha256")
        != request.density_qc_policy_sha256
        or state.get("density_qc_implementation_sha256")
        != request.density_qc_implementation_sha256
        or state.get("qc_implementation_sha256")
        != request.density_qc_implementation_sha256
    ):
        return None
    provenance_ref = state.get("state_point_provenance")
    if not isinstance(provenance_ref, dict):
        return None
    declared_path = provenance_ref.get("path")
    declared_bytes = provenance_ref.get("bytes")
    if not isinstance(declared_path, str):
        return None
    linked_provenance = (run_root / declared_path).resolve()
    if (
        linked_provenance != candidate.provenance_path.resolve()
        or provenance_ref.get("sha256") != sha256_file(candidate.provenance_path)
        or isinstance(declared_bytes, bool)
        or not isinstance(declared_bytes, int)
        or declared_bytes != candidate.provenance_path.stat().st_size
    ):
        return None
    artifacts = state.get("artifacts")
    thermo_ref = artifacts.get("thermo_samples") if isinstance(artifacts, dict) else None
    qc_ref = artifacts.get("state_point_qc") if isinstance(artifacts, dict) else None
    if not isinstance(thermo_ref, dict):
        return None
    thermo_path_value = thermo_ref.get("path")
    thermo_bytes = thermo_ref.get("bytes")
    if not isinstance(thermo_path_value, str) or not thermo_path_value:
        return None
    manifest_thermo_path = (run_root / thermo_path_value).resolve()
    try:
        manifest_thermo_path.relative_to(run_root)
    except ValueError:
        return None
    if (
        manifest_thermo_path != candidate.artifact_path.resolve()
        or thermo_ref.get("sha256") != artifact.sha256
        or isinstance(thermo_bytes, bool)
        or not isinstance(thermo_bytes, int)
        or thermo_bytes != artifact.size_bytes
        or not manifest_thermo_path.is_file()
        or manifest_thermo_path.stat().st_size != artifact.size_bytes
        or sha256_file(manifest_thermo_path) != artifact.sha256
    ):
        return None
    if not isinstance(qc_ref, dict):
        return None
    qc_path_value = qc_ref.get("path")
    qc_bytes = qc_ref.get("bytes")
    if not isinstance(qc_path_value, str) or not qc_path_value:
        return None
    manifest_qc_path = (run_root / qc_path_value).resolve()
    try:
        manifest_qc_path.relative_to(run_root)
    except ValueError:
        return None
    if (
        manifest_qc_path != density_qc_path.resolve()
        or qc_ref.get("sha256") != density_qc_artifact.sha256
        or isinstance(qc_bytes, bool)
        or not isinstance(qc_bytes, int)
        or qc_bytes != density_qc_artifact.size_bytes
        or not manifest_qc_path.is_file()
        or manifest_qc_path.stat().st_size != density_qc_artifact.size_bytes
        or sha256_file(manifest_qc_path) != density_qc_artifact.sha256
    ):
        return None
    return candidate


def select_reuse_candidate(
    records: Iterable[tuple[str | Path, StatePointProvenance]],
    request: ReuseRequest,
) -> ReuseDecision:
    """Return run-new/reuse for zero/one matches; fail on multiple matches.

    No ordering, timestamps, prompts, or "latest" heuristic are used.  Repeated
    references to the same provenance file are deduplicated so overlapping
    search roots do not create a false ambiguity.
    """

    eligible_by_path: dict[Path, ReuseCandidate] = {}
    for raw_path, record in records:
        provenance_path = Path(raw_path).resolve()
        candidate = _eligible_candidate(provenance_path, record, request)
        if candidate is not None:
            eligible_by_path.setdefault(provenance_path, candidate)

    eligible = [eligible_by_path[path] for path in sorted(eligible_by_path, key=str)]
    if not eligible:
        return ReuseDecision(
            action=ReuseAction.RUN_NEW,
            candidate=None,
            reason="no exact, status-passing, hash-verified reuse candidate",
        )
    if len(eligible) > 1:
        paths = ", ".join(
            f"{candidate.provenance.run_id}/{candidate.provenance.state_point_id} "
            f"({candidate.provenance_path})"
            for candidate in eligible
        )
        raise AmbiguousReuseError(
            f"multiple exact reusable candidates for replica {request.replica_id}: {paths}"
        )
    return ReuseDecision(
        action=ReuseAction.REUSE,
        candidate=eligible[0],
        reason="one exact, status-passing, hash-verified reuse candidate",
    )


def resolve_cross_run_reuse(
    search_roots: Sequence[str | Path],
    request: ReuseRequest,
    *,
    provenance_filename: str = PROVENANCE_FILENAME,
) -> ReuseDecision:
    """Discover provenance records below roots and resolve reuse conservatively."""

    discovered: dict[Path, StatePointProvenance] = {}
    for raw_root in search_roots:
        root = Path(raw_root)
        if not root.exists():
            continue
        if root.is_file():
            paths = [root] if root.name == provenance_filename else []
        else:
            paths = root.rglob(provenance_filename)
        for path in paths:
            resolved = path.resolve()
            if any(part.endswith(".incomplete") for part in resolved.parts):
                continue
            if resolved not in discovered:
                # A malformed provenance document is a fail-closed error, not a
                # silently ignored candidate.
                discovered[resolved] = read_state_point_provenance(resolved)
    return select_reuse_candidate(discovered.items(), request)
