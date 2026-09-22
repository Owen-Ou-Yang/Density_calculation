"""Offline integrity and disclosure check for the sealed SMILES/code release."""
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / "PUBLIC_MANIFEST.json").read_text())
    expected = {r["path"]: r for r in manifest["files"]}
    if len(expected) != len(manifest["files"]):
        raise ValueError("duplicate path in public manifest")
    actual = set()
    patterns = ["/" + "Users" + r"/[^/\s]+", "/" + "users" + r"/[^/\s]+",
                r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
                r"\bgh[pousr]_[A-Za-z0-9]{20,}", r"\bhf_[A-Za-z0-9]{20,}\b"]
    forbidden = {".data", ".lmps", ".xyz", ".mol", ".pt", ".pth", ".model", ".pkl", ".log", ".restart", ".dump", ".xlsx"}
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if set(rel.parts) & {".git", "__pycache__", ".pytest_cache"}:
            continue
        if path.is_symlink():
            raise ValueError(f"symlink: {rel}")
        if not path.is_file() or rel.as_posix() == "PUBLIC_MANIFEST.json":
            continue
        name = rel.as_posix()
        if path.suffix in forbidden or (path.suffix == ".csv" and name != "inputs/smiles.csv"):
            raise ValueError(f"nonpublic artifact: {rel}")
        if any(re.search(pattern, path.read_text()) for pattern in patterns):
            raise ValueError(f"potential private path/credential: {rel}")
        if name not in expected or sha(path) != expected[name]["sha256"] or path.stat().st_size != expected[name]["bytes"]:
            raise ValueError(f"unexpected or changed file: {rel}")
        actual.add(name)
    if actual != set(expected):
        raise ValueError("missing files")
    identity = json.loads((root / "inputs/catalog_identity.json").read_text())
    catalog = root / "inputs/smiles.csv"
    if sha(catalog) != identity["catalog_sha256"]:
        raise ValueError("catalog byte hash mismatch")
    with catalog.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["task_id", "smiles", "smiles_sha256"]:
            raise ValueError("only ID, original SMILES and its hash are allowed")
        rows = list(reader)
    if len(rows) != 1691 or len({r["smiles"] for r in rows}) != 1691:
        raise ValueError("1691 exact strings required")
    for index, row in enumerate(rows, 1):
        if row["task_id"] != f"S{index:06d}" or hashlib.sha256(row["smiles"].encode()).hexdigest() != row["smiles_sha256"]:
            raise ValueError("SMILES identity mismatch")
    return {"status": "PASS", "files": len(actual)+1, "smiles": len(rows),
            "experimental_values_included": False, "starting_structures_included": False,
            "model_execution": False, "network_access": False}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
