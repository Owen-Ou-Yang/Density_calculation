"""Disk-verified CPU preparation handoff; no chemistry/model imports or execution."""

from pathlib import Path


def _safe(path, root):
    path, root = Path(path), Path(root)
    if not path.is_absolute() or not root.is_absolute():
        raise ValueError("preparation evidence paths must be absolute")
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ValueError(f"preparation evidence symlink rejected: {candidate}")
    from .cli import _contained
    return _contained(path, root)


def preparation_evidence(prepared_path, task, attempt):
    """Re-read final QC, producer stage records and their bound bytes on disk."""
    from .cli import _artifact, _identity, _object, _prepared_inputs
    from .catalog import task_seed
    prepared_path, attempt = Path(prepared_path), Path(attempt)
    prepared = _object(_safe(prepared_path, attempt))
    _identity(prepared, task)
    provenance = prepared.get("preparation_provenance", {})
    if (not isinstance(provenance, dict)
            or provenance.get("original_smiles") != task["smiles"]
            or provenance.get("smiles_sha256") != task["smiles_sha256"]
            or provenance.get("seed") != task_seed(task["smiles_sha256"])
            or provenance.get("classical_equilibrium_check") is not True):
        raise ValueError("preparation lacks same-SMILES/seed classical QC PASS provenance")
    for label in ("input_data", "snapshot_metadata"):
        reference = prepared.get(label)
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"prepared manifest requires {label}")
        path = Path(reference)
        _safe(path if path.is_absolute() else prepared_path.parent / path, attempt)
    inputs = _prepared_inputs(prepared_path, task, attempt)
    for record in inputs.values():
        _safe(Path(record["path"]), attempt)
    metadata = _object(Path(inputs["snapshot_metadata"]["path"]))
    if (metadata.get("snapshot_sha256") != inputs["input_data"]["sha256"]
            or metadata.get("snapshot_bytes") != inputs["input_data"]["bytes"]):
        raise ValueError("preparation snapshot metadata does not bind input bytes")
    history = provenance.get("equilibration_history")
    if (not isinstance(history, list) or not history
            or not all(isinstance(item, dict) for item in history)
            or history[-1].get("classical_equilibrium_check") is not True
            or history[-1].get("status") != "QC_PASS"):
        raise ValueError("preparation has no final classical QC PASS history")
    records, indices = {}, set()
    for stage in history:
        index = stage.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 3 or index in indices:
            raise ValueError("invalid or duplicate classical stage identity")
        indices.add(index)
        label = f"preparation_stage_eq{index:04d}"
        path = _safe(attempt / "prep" / f"equilibration_stage_eq{index:04d}.json", attempt)
        if _object(path) != stage:
            raise ValueError("classical history differs from its completed on-disk stage")
        if (stage.get("status") not in {"QC_PASS", "QC_NOT_CONVERGED"}
                or not stage.get("ended_utc")
                or not isinstance(stage.get("lammps_returncodes"), list)
                or not stage["lammps_returncodes"]
                or any(type(code) is not int or code != 0 for code in stage["lammps_returncodes"])):
            raise ValueError("classical stage lacks successful terminal engine exits")
        records[label] = _artifact(path)
        artifacts = stage.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("classical stage lacks byte-bound artifacts")
        paths = set()
        for record in artifacts:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise ValueError("invalid classical stage artifact record")
            actual_path = _safe(Path(record["path"]), attempt)
            if str(actual_path) in paths or _artifact(actual_path) != record:
                raise ValueError("classical stage artifact duplicate/hash mismatch")
            paths.add(str(actual_path))
        if not isinstance(stage.get("final_data"), str):
            raise ValueError("classical stage final data path must be a string")
        final = _safe(Path(stage["final_data"]), attempt)
        final_record = _artifact(final)
        if str(final) not in paths or final_record["sha256"] != stage.get("final_data_sha256"):
            raise ValueError("classical final data not bound by completed stage")
    if Path(history[-1]["final_data"]).resolve() != Path(inputs["input_data"]["path"]):
        raise ValueError("prepared input is not the final QC-passing classical state")
    return records


def prepared_parent_selector(attempt_path, task):
    """Accept only an immutable terminal CPU attempt, never an in-memory claim."""
    from .cli import RECEIPT_SCHEMA, _artifact, _identity, _object, _prepared_inputs
    from .catalog import task_seed
    attempt = Path(attempt_path)
    if not attempt.is_absolute():
        raise ValueError("prepared parent attempt must be absolute")
    for path in (attempt, *attempt.parents):
        if path.is_symlink():
            raise ValueError("prepared parent attempt cannot contain symlinks")
    attempt = attempt.resolve()
    receipt_path = _safe(attempt / "receipt.json", attempt)
    receipt_record = _artifact(receipt_path)
    receipt = _object(receipt_path)
    _identity(receipt, task)
    if (receipt.get("schema_version") != RECEIPT_SCHEMA
            or receipt.get("smiles") != task["smiles"]
            or receipt.get("attempt_id") != attempt.name
            or receipt.get("attempt_root") != str(attempt)
            or receipt.get("execution_stage") != "prepare"
            or receipt.get("status") != "PREPARED_QC_PASS"
            or receipt.get("pipeline_complete") is not False
            or receipt.get("classical_qc_status") != "PASS"
            or receipt.get("density_g_cm3") is not None
            or not receipt.get("ended_at")):
        raise ValueError("parent is not a terminal PREPARED_QC_PASS CPU attempt")
    stages = receipt.get("stages")
    if (not isinstance(stages, list) or len(stages) != 1
            or not isinstance(stages[0], dict)
            or stages[0].get("stage") != "prepare"
            or type(stages[0].get("returncode")) is not int or stages[0]["returncode"] != 0
            or not stages[0].get("ended_at")):
        raise ValueError("CPU parent lacks exactly one successful preparation process")
    artifacts = receipt.get("artifacts")
    fixed = {"request": "request.json", "prepared": "prep/prepared.json",
             "prepare_stdout": "prepare.stdout.log", "prepare_stderr": "prepare.stderr.log"}
    if not isinstance(artifacts, dict) or not set(fixed).issubset(artifacts):
        raise ValueError("CPU parent lacks required artifact records")
    for label, record in artifacts.items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("invalid CPU parent artifact record")
        path = _safe(Path(record["path"]), attempt)
        if _artifact(path) != record:
            raise ValueError(f"CPU parent artifact changed: {label}")
    for label, relative in fixed.items():
        if artifacts[label]["path"] != str(attempt / relative):
            raise ValueError("CPU parent fixed artifact path changed")
    request = _object(attempt / "request.json")
    _identity(request, task)
    if (request.get("schema_version") != "polymer-smiles-task/v1"
            or request.get("smiles") != task["smiles"]
            or request.get("seed") != task_seed(task["smiles_sha256"])
            or request.get("attempt_id") != attempt.name
            or request.get("attempt_root") != str(attempt)
            or request.get("execution_stage") != "prepare"
            or request.get("prepared_manifest_path") != str(attempt / "prep/prepared.json")):
        raise ValueError("CPU parent request identity changed")
    prepared = attempt / "prep/prepared.json"
    expected = {**_prepared_inputs(prepared, task, attempt),
                **preparation_evidence(prepared, task, attempt)}
    if set(artifacts) != set(fixed) | set(expected):
        raise ValueError("CPU parent artifact set does not match verified preparation")
    if any(artifacts.get(label) != record for label, record in expected.items()):
        raise ValueError("CPU parent does not bind verified preparation evidence")
    if _artifact(receipt_path) != receipt_record:
        raise ValueError("CPU parent receipt changed during verification")
    return {"attempt_root": str(attempt), "artifacts": {**artifacts, "receipt": receipt_record}}


def copy_prepared_parent(selector, task, attempt, receipt):
    from .cli import _atomic_json, _copy_parent_preparation, _copy_verified_record
    # Recheck at actual intake, not just when a scheduler plan was rendered.
    if prepared_parent_selector(Path(selector["attempt_root"]), task) != selector:
        raise ValueError("CPU parent changed after selection")
    _copy_parent_preparation({"smiles_evidence": selector["artifacts"]}, task, attempt, receipt)
    evidence_dir = attempt / "prep/parent_evidence"
    for label, record in selector["artifacts"].items():
        if label in {"request", "receipt", "prepared", "input_data", "snapshot_metadata"}:
            continue
        suffix = ".log" if label.startswith("prepare_") else ".json"
        receipt["artifacts"]["parent_" + label] = _copy_verified_record(record, evidence_dir / (label + suffix))
    receipt["preparation_action"] = "COPIED_VERIFIED_CPU_PARENT_INPUTS"
    receipt["prepared_parent_receipt_sha256"] = selector["artifacts"]["receipt"]["sha256"]
    _atomic_json(attempt / "receipt.json", receipt)
