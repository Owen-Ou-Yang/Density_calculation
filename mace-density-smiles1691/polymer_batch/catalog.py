"""Read the exact original strings without chemical rewriting or normalization."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any


CATALOG_SIZE = 1691


def smiles_digest(smiles: str) -> str:
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()


def task_seed(smiles_sha256: str) -> int:
    return int(smiles_sha256[:16], 16) % 899999999 + 1


def load_catalog(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["task_id", "smiles", "smiles_sha256"]:
            raise ValueError("catalog columns must be exactly task_id,smiles,smiles_sha256")
        rows = list(reader)
    if len(rows) != CATALOG_SIZE:
        raise ValueError(f"catalog must contain exactly {CATALOG_SIZE} entries")
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        if row.get("task_id") != f"S{index:06d}":
            raise ValueError(f"catalog row {index} has an unexpected task_id")
        smiles = row.get("smiles")
        if not isinstance(smiles, str) or not smiles:
            raise ValueError(f"catalog row {index} has an empty SMILES")
        if smiles in seen:
            raise ValueError(f"catalog row {index} duplicates an exact SMILES")
        seen.add(smiles)
        if row.get("smiles_sha256") != smiles_digest(smiles):
            raise ValueError(f"catalog row {index} does not match its exact-string SHA-256")
        if None in row:
            raise ValueError(f"catalog row {index} contains extra columns")
    return rows


def select_tasks(rows, *, task_index=None, start=None, stop=None, shard_index=None, shard_count=None):
    """Ranges are inclusive and one based; sharding is zero based modulo order."""
    groups = int(task_index is not None) + int(start is not None or stop is not None) + int(
        shard_index is not None or shard_count is not None
    )
    if groups > 1:
        raise ValueError("choose a task index, an inclusive range, or a shard")
    if task_index is not None:
        if not 1 <= task_index <= len(rows):
            raise ValueError("task-index is outside the catalog")
        return [rows[task_index - 1]]
    if shard_index is not None or shard_count is not None:
        if shard_index is None or shard_count is None or shard_count < 1 or not 0 <= shard_index < shard_count:
            raise ValueError("shards require shard-count >= 1 and 0 <= shard-index < shard-count")
        return [row for index, row in enumerate(rows) if index % shard_count == shard_index]
    first = 1 if start is None else start
    last = len(rows) if stop is None else stop
    if not 1 <= first <= last <= len(rows):
        raise ValueError("start/stop must form an inclusive one-based catalog range")
    return rows[first - 1:last]
