"""Authenticated input-snapshot contracts for thermal-property campaigns.

The LAMMPS data file alone cannot say whether it is a dilute APG packing, an
ADEPT eq2 structure, or a structure already relaxed with MACE.  This module
keeps that distinction explicit and binds the declaration to the exact bytes
that will be simulated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .provenance import canonical_sha256, sha256_file


SNAPSHOT_CONTRACT_SCHEMA = "thermal-properties-snapshot-contract/v1"


class SnapshotContractError(ValueError):
    """Raised when snapshot provenance is missing, inconsistent, or unsafe."""


class SnapshotClass(str, Enum):
    """The preparation level of one polymer structure."""

    APG_RAW = "APG_RAW"
    CLASSICAL_EQ2 = "CLASSICAL_EQ2"
    MACE_EQUILIBRATED = "MACE_EQUILIBRATED"


_SOURCE_METHOD_BY_CLASS = {
    SnapshotClass.APG_RAW: {"ADEPT_APG"},
    # Public SMILES batch: retain the real collaborator preparation provenance.
    # This adds an input method, not a change to MD or QC numerical settings.
    SnapshotClass.CLASSICAL_EQ2: {"ADEPT_EQ1_EQ2", "RADONPY_EQ21"},
    SnapshotClass.MACE_EQUILIBRATED: {"MACE_TRANSITION_NPT"},
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class SnapshotContract:
    snapshot_path: Path
    metadata_path: Path
    snapshot_class: SnapshotClass
    source_method: str
    packing_id: str
    snapshot_sha256: str
    snapshot_bytes: int
    metadata_sha256: str
    metadata_bytes: int
    atom_count: int
    atom_type_count: int
    element_counts: Mapping[str, int]
    payload_sha256: str
    mace_model_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SNAPSHOT_CONTRACT_SCHEMA,
            "snapshot_path": str(self.snapshot_path),
            "metadata_path": str(self.metadata_path),
            "snapshot_class": self.snapshot_class.value,
            "source_method": self.source_method,
            "packing_id": self.packing_id,
            "snapshot_sha256": self.snapshot_sha256,
            "snapshot_bytes": self.snapshot_bytes,
            "metadata_sha256": self.metadata_sha256,
            "metadata_bytes": self.metadata_bytes,
            "atom_count": self.atom_count,
            "atom_type_count": self.atom_type_count,
            "element_counts": dict(sorted(self.element_counts.items())),
            "payload_sha256": self.payload_sha256,
            "mace_model_sha256": self.mace_model_sha256,
        }


def _safe_file(path: str | Path, field: str) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise SnapshotContractError(f"{field} must be an absolute path")
    if raw.is_symlink():
        raise SnapshotContractError(f"{field} cannot be a symlink: {raw}")
    resolved = raw.resolve()
    if not resolved.is_file():
        raise SnapshotContractError(f"{field} is missing or unsafe: {resolved}")
    return resolved


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SnapshotContractError(f"{field} must be a positive integer")
    return value


def _declared_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.lower()):
        raise SnapshotContractError(f"{field} must be a 64-character SHA-256")
    return value.lower()


def lammps_data_counts(path: str | Path) -> tuple[int, int]:
    """Read only the LAMMPS data header atom and atom-type counts."""

    source = Path(path)
    atoms: int | None = None
    atom_types: int | None = None
    try:
        with source.open("r", encoding="utf-8", errors="replace") as handle:
            for index, raw_line in enumerate(handle):
                if index > 256:
                    break
                line = raw_line.split("#", 1)[0].strip()
                atom_match = re.fullmatch(r"([0-9]+)\s+atoms", line)
                type_match = re.fullmatch(r"([0-9]+)\s+atom\s+types", line)
                if atom_match:
                    atoms = int(atom_match.group(1))
                if type_match:
                    atom_types = int(type_match.group(1))
                if atoms is not None and atom_types is not None:
                    break
    except OSError as exc:
        raise SnapshotContractError(f"cannot read LAMMPS data header: {source}") from exc
    if atoms is None or atoms <= 0 or atom_types is None or atom_types <= 0:
        raise SnapshotContractError(
            f"LAMMPS data header lacks positive atoms/atom types counts: {source}"
        )
    return atoms, atom_types


def build_snapshot_contract_payload(
    *,
    snapshot_path: str | Path,
    snapshot_class: str | SnapshotClass,
    packing_id: str,
    element_counts: Mapping[str, int],
    mace_model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build, but do not write, a contract for an existing snapshot."""

    snapshot = _safe_file(snapshot_path, "snapshot")
    try:
        declared_class = SnapshotClass(snapshot_class)
    except ValueError as exc:
        raise SnapshotContractError("unsupported snapshot_class") from exc
    if not isinstance(packing_id, str) or not _IDENTIFIER_RE.fullmatch(packing_id):
        raise SnapshotContractError(
            "packing_id must match [A-Za-z0-9][A-Za-z0-9_.-]*"
        )
    if not isinstance(element_counts, Mapping) or not element_counts:
        raise SnapshotContractError("element_counts must be a nonempty object")
    normalized_counts: dict[str, int] = {}
    for element, count in element_counts.items():
        if not isinstance(element, str) or not re.fullmatch(r"[A-Z][a-z]?", element):
            raise SnapshotContractError(f"invalid element_counts key: {element!r}")
        normalized_counts[element] = _positive_int(count, f"element_counts.{element}")
    atom_count, atom_type_count = lammps_data_counts(snapshot)
    if sum(normalized_counts.values()) != atom_count:
        raise SnapshotContractError("element_counts do not sum to atom_count")
    payload: dict[str, Any] = {
        "schema_version": SNAPSHOT_CONTRACT_SCHEMA,
        "snapshot_class": declared_class.value,
        "source_method": ("ADEPT_EQ1_EQ2" if declared_class is SnapshotClass.CLASSICAL_EQ2
                          else next(iter(_SOURCE_METHOD_BY_CLASS[declared_class]))),
        "packing_id": packing_id,
        "snapshot_sha256": sha256_file(snapshot),
        "snapshot_bytes": snapshot.stat().st_size,
        "atom_count": atom_count,
        "atom_type_count": atom_type_count,
        "element_counts": dict(sorted(normalized_counts.items())),
    }
    if declared_class is SnapshotClass.MACE_EQUILIBRATED:
        if mace_model_path is None:
            raise SnapshotContractError(
                "MACE_EQUILIBRATED requires an exact mace_model_path"
            )
        model = _safe_file(mace_model_path, "MACE model")
        payload["mace_model_sha256"] = sha256_file(model)
    elif mace_model_path is not None:
        raise SnapshotContractError(
            "mace_model_path is valid only for MACE_EQUILIBRATED"
        )
    return payload


def load_snapshot_contract(
    *,
    snapshot_path: str | Path,
    metadata_path: str | Path,
    expected_class: str | SnapshotClass,
    mace_model_sha256: str | None = None,
) -> SnapshotContract:
    """Validate a detached contract against exact snapshot and model bytes."""

    snapshot = _safe_file(snapshot_path, "snapshot")
    metadata = _safe_file(metadata_path, "snapshot metadata")
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotContractError(f"cannot read snapshot metadata: {metadata}") from exc
    if not isinstance(payload, dict):
        raise SnapshotContractError("snapshot metadata root must be an object")
    required = {
        "schema_version",
        "snapshot_class",
        "source_method",
        "packing_id",
        "snapshot_sha256",
        "snapshot_bytes",
        "atom_count",
        "atom_type_count",
        "element_counts",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise SnapshotContractError(
            f"snapshot metadata is missing required fields: {missing}"
        )
    if payload.get("schema_version") != SNAPSHOT_CONTRACT_SCHEMA:
        raise SnapshotContractError(
            f"snapshot metadata schema must be {SNAPSHOT_CONTRACT_SCHEMA!r}"
        )
    try:
        declared_class = SnapshotClass(payload.get("snapshot_class"))
        required_class = SnapshotClass(expected_class)
    except ValueError as exc:
        raise SnapshotContractError("unsupported snapshot_class") from exc
    if declared_class is not required_class:
        raise SnapshotContractError(
            "snapshot metadata class disagrees with the thermal request"
        )
    source_method = payload.get("source_method")
    if source_method not in _SOURCE_METHOD_BY_CLASS[declared_class]:
        raise SnapshotContractError(
            f"source_method {source_method!r} is invalid for {declared_class.value}"
        )
    packing_id = payload.get("packing_id")
    if not isinstance(packing_id, str) or not _IDENTIFIER_RE.fullmatch(packing_id):
        raise SnapshotContractError(
            "packing_id must match [A-Za-z0-9][A-Za-z0-9_.-]*"
        )
    declared_snapshot_sha256 = _declared_sha256(
        payload.get("snapshot_sha256"), "snapshot_sha256"
    )
    actual_snapshot_sha256 = sha256_file(snapshot)
    if declared_snapshot_sha256 != actual_snapshot_sha256:
        raise SnapshotContractError("snapshot bytes do not match snapshot metadata")
    declared_bytes = _positive_int(payload.get("snapshot_bytes"), "snapshot_bytes")
    if declared_bytes != snapshot.stat().st_size:
        raise SnapshotContractError("snapshot size does not match snapshot metadata")
    atom_count = _positive_int(payload.get("atom_count"), "atom_count")
    atom_type_count = _positive_int(payload.get("atom_type_count"), "atom_type_count")
    header_atom_count, header_atom_types = lammps_data_counts(snapshot)
    if atom_count != header_atom_count or atom_type_count != header_atom_types:
        raise SnapshotContractError(
            "snapshot atom counts disagree with the authenticated LAMMPS data header"
        )
    element_counts = payload.get("element_counts")
    if not isinstance(element_counts, Mapping) or not element_counts:
        raise SnapshotContractError("element_counts must be a nonempty object")
    normalized_counts: dict[str, int] = {}
    for element, count in element_counts.items():
        if not isinstance(element, str) or not re.fullmatch(r"[A-Z][a-z]?", element):
            raise SnapshotContractError(f"invalid element_counts key: {element!r}")
        normalized_counts[element] = _positive_int(count, f"element_counts.{element}")
    if sum(normalized_counts.values()) != atom_count:
        raise SnapshotContractError("element_counts do not sum to atom_count")

    declared_model = payload.get("mace_model_sha256")
    normalized_model: str | None = None
    if declared_class is SnapshotClass.MACE_EQUILIBRATED:
        normalized_model = _declared_sha256(
            declared_model, "mace_model_sha256"
        )
        if mace_model_sha256 is None or normalized_model != mace_model_sha256:
            raise SnapshotContractError(
                "MACE-equilibrated snapshot was prepared with a different model"
            )
    elif declared_model is not None:
        normalized_model = _declared_sha256(declared_model, "mace_model_sha256")

    return SnapshotContract(
        snapshot_path=snapshot,
        metadata_path=metadata,
        snapshot_class=declared_class,
        source_method=str(source_method),
        packing_id=packing_id,
        snapshot_sha256=actual_snapshot_sha256,
        snapshot_bytes=snapshot.stat().st_size,
        metadata_sha256=sha256_file(metadata),
        metadata_bytes=metadata.stat().st_size,
        atom_count=atom_count,
        atom_type_count=atom_type_count,
        element_counts=normalized_counts,
        payload_sha256=canonical_sha256(payload),
        mace_model_sha256=normalized_model,
    )
