"""Small standard-library dispatcher; model/preparation adapters are external.

Only run --confirm-run YES launches child processes.  A task has one atomic
POSIX file lock and immutable numbered attempts.  No attempt is overwritten or
implicitly retried.  Receipts keep process completion separate from QC success.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

from .catalog import load_catalog, select_tasks, task_seed
from .preparation_handoff import (
    preparation_evidence as _preparation_evidence,
    prepared_parent_selector,
    copy_prepared_parent as _copy_prepared_parent,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_SCHEMA = "polymer-smiles-attempt-receipt/v1"
COMPLETE_STATUSES = {"COMPLETE_QC_PASS", "COMPLETE_QC_FAIL"}
SAMPLING_STATUSES = {"NEEDS_MORE_SAMPLING", "SAMPLING_BUDGET_EXHAUSTED", "SAMPLING_QC_FAILED"}
_ATTEMPT = re.compile(r"attempt_([0-9]{4,})")
# The density core owns a separate LAMMPS session and may spend 5 seconds on
# TERM plus 2 seconds on KILL. Let its adapter finish that nested cleanup.
_ADAPTER_TERMINATION_GRACE_SECONDS = 15


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _artifact(path: Path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact is not a regular, non-symlink file: {path}")
    return {"path": str(path.resolve()), "sha256": _sha(path), "bytes": path.stat().st_size}


def _contained(path: Path, root: Path):
    resolved = path.resolve()
    if resolved == root.resolve() or root.resolve() not in resolved.parents:
        raise ValueError(f"artifact escapes its attempt directory: {path}")
    return resolved


def _site(path: Path, stage="all"):
    if not path.is_absolute():
        raise ValueError("--site must be an absolute JSON path")
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("site JSON must be an object")
    if stage not in {"all", "prepare", "density"}:
        raise ValueError("unknown dispatch stage")
    if stage == "prepare":
        preparation = value.get("preparation", {})
        gpu = preparation.get("gpu", 0) if isinstance(preparation, dict) else None
        if type(gpu) is not int or gpu != 0:
            raise ValueError("CPU-only prepare stage requires preparation.gpu=0")
    for name in (("prepare_argv", "density_argv") if stage == "all" else (stage + "_argv",)):
        argv = value.get(name)
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
            raise ValueError(f"site.{name} must be a nonempty argv array")
        for placeholder in ("{request}", "{output_dir}", "{site}"):
            if not any(placeholder in token for token in argv):
                raise ValueError(f"site.{name} is missing {placeholder}")
    value["_batch_site_sha256"] = hashlib.sha256(raw).hexdigest()
    return value


def _argv(template, request: Path, output_dir: Path, site: Path):
    replacements = {"{request}": str(request), "{output_dir}": str(output_dir), "{site}": str(site)}
    result = []
    for token in template:
        for marker, value in replacements.items():
            token = token.replace(marker, value)
        result.append(token)
    return result


def _identity(value, task):
    if value.get("task_id") != task["task_id"] or value.get("smiles_sha256") != task["smiles_sha256"]:
        raise ValueError("artifact task_id or exact SMILES identity does not match its request")


def _prepared_inputs(prepared: Path, task, attempt: Path):
    value = _object(prepared)
    _identity(value, task)
    records = {}
    for key in ("input_data", "snapshot_metadata"):
        reference = value.get(key)
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"prepared manifest requires {key}")
        path = Path(reference)
        if not path.is_absolute():
            path = prepared.parent / path
        records[key] = _artifact(_contained(path, attempt))
    return records


def _result(path: Path, task, attempt: Path):
    value = _object(path)
    _identity(value, task)
    if "smiles" in value and value["smiles"] != task["smiles"]:
        raise ValueError("density result changed the exact original SMILES")
    if "attempt_id" in value and value["attempt_id"] != attempt.name:
        raise ValueError("density result refers to a different attempt")
    if value.get("execution_status") != "COMPLETE":
        raise ValueError("density result does not report execution_status=COMPLETE")
    if value.get("qc_status") not in {"PASS", "FAIL", "NOT_EVALUATED"}:
        raise ValueError("density result requires an explicit PASS, FAIL, or NOT_EVALUATED qc_status")
    reference = value.get("run_manifest")
    if not isinstance(reference, str) or not reference:
        raise ValueError("density result requires a run_manifest path")
    manifest = Path(reference)
    if not manifest.is_absolute():
        manifest = path.parent / manifest
    manifest = _contained(manifest, attempt)
    _artifact(manifest)
    for key, artifact in (("request_sha256", attempt / "request.json"),
                          ("prepared_manifest_sha256", attempt / "prep" / "prepared.json"),
                          ("run_manifest_sha256", manifest)):
        if key in value and value[key] != _sha(artifact):
            raise ValueError(f"density result declared an inconsistent {key}")
    for key in ("density_g_cm3", "density_standard_error_g_cm3"):
        number = value.get(key)
        if number is not None and (isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0):
            raise ValueError(f"invalid optional density summary: {key}")
    return value, manifest


def _attempts(task_root: Path):
    if not task_root.exists():
        return []
    found = []
    for path in task_root.iterdir():
        match = _ATTEMPT.fullmatch(path.name)
        if match:
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"unsafe attempt path: {path}")
            found.append((int(match.group(1)), path))
    return sorted(found)


def _verify_complete(receipt, task, attempt: Path):
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("unsupported attempt receipt")
    _identity(receipt, task)
    if receipt.get("smiles") != task["smiles"]:
        raise ValueError("receipt original SMILES string has changed")
    artifacts = receipt.get("artifacts")
    required = {"request", "prepared", "input_data", "snapshot_metadata", "result", "run_manifest", "prepare_stdout", "prepare_stderr", "density_stdout", "density_stderr"}
    if not isinstance(artifacts, dict) or not required.issubset(artifacts):
        raise ValueError("complete receipt lacks required artifact hashes")
    for record in artifacts.values():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("invalid artifact record")
        path = _contained(Path(record["path"]), attempt)
        if _artifact(path) != record:
            raise ValueError(f"completed artifact hash or size changed: {path}")
    request = _object(attempt / "request.json")
    _identity(request, task)
    if request.get("schema_version") != "polymer-smiles-task/v1" or request.get("smiles") != task["smiles"]:
        raise ValueError("completed request does not preserve the original SMILES")
    if request.get("seed") != task_seed(task["smiles_sha256"]) or request.get("attempt_id") != attempt.name:
        raise ValueError("completed request seed or attempt identity changed")
    if request.get("attempt_root") != str(attempt.resolve()) or request.get("prepared_manifest_path") != str((attempt / "prep" / "prepared.json").resolve()):
        raise ValueError("completed request attempt paths changed")
    prepared_inputs = _prepared_inputs(attempt / "prep" / "prepared.json", task, attempt)
    if any(artifacts[key] != record for key, record in prepared_inputs.items()):
        raise ValueError("receipt does not bind the prepared input files")
    result, manifest = _result(attempt / "density" / "result.json", task, attempt)
    fixed_paths = {"request": "request.json", "prepared": "prep/prepared.json", "result": "density/result.json",
                   "prepare_stdout": "prepare.stdout.log", "prepare_stderr": "prepare.stderr.log",
                   "density_stdout": "density.stdout.log", "density_stderr": "density.stderr.log"}
    if any(artifacts[label]["path"] != str((attempt / relative).resolve()) for label, relative in fixed_paths.items()):
        raise ValueError("receipt does not bind the expected task artifact files")
    if artifacts["run_manifest"]["path"] != str(manifest):
        raise ValueError("receipt does not bind the declared run manifest")
    expected = "COMPLETE_QC_PASS" if result["qc_status"] == "PASS" else "COMPLETE_QC_FAIL"
    if receipt.get("status") != expected:
        raise ValueError("receipt completion status disagrees with density QC")
    return result


def _summary(task, task_root: Path):
    attempts = _attempts(task_root)
    base = {"task_id": task["task_id"], "smiles_sha256": task["smiles_sha256"]}
    if not attempts:
        return {**base, "status": "NOT_STARTED"}
    attempt = attempts[-1][1]
    base["attempt_id"] = attempt.name
    base["attempt_root"] = str(attempt)
    try:
        receipt = _object(attempt / "receipt.json")
        _identity(receipt, task)
        status = receipt.get("status")
        if status == "PREPARED_QC_PASS":
            prepared_parent_selector(attempt, task)
            return {**base, "status": status, "integrity": "VERIFIED",
                    "execution_stage": "prepare", "classical_qc_status": "PASS",
                    "pipeline_complete": False, "density_g_cm3": None}
        if status in COMPLETE_STATUSES:
            result = _verify_complete(receipt, task, attempt)
            base.update({key: result.get(key) for key in ("execution_status", "qc_status", "density_g_cm3", "density_standard_error_g_cm3")})
            return {**base, "status": status, "integrity": "VERIFIED"}
        if status in SAMPLING_STATUSES:
            artifacts = receipt.get("artifacts")
            required = {"request", "prepared", "input_data", "snapshot_metadata", "sampling_outcome",
                        "prepare_stdout", "prepare_stderr", "density_stdout", "density_stderr"}
            if (receipt.get("schema_version") != RECEIPT_SCHEMA
                    or receipt.get("smiles") != task["smiles"]
                    or receipt.get("attempt_id") != attempt.name
                    or receipt.get("attempt_root") != str(attempt.resolve())
                    or not receipt.get("ended_at")
                    or not isinstance(artifacts, dict) or not required.issubset(artifacts)):
                raise ValueError("sampling receipt lacks terminal identity or required artifacts")
            for record in artifacts.values():
                if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                    raise ValueError("invalid sampling artifact record")
                path = _contained(Path(record["path"]), attempt)
                if _artifact(path) != record:
                    raise ValueError("sampling receipt artifact changed")
            fixed_paths = {"request": "request.json", "prepared": "prep/prepared.json",
                           "sampling_outcome": "density/sampling_outcome.json",
                           "prepare_stdout": "prepare.stdout.log", "prepare_stderr": "prepare.stderr.log",
                           "density_stdout": "density.stdout.log", "density_stderr": "density.stderr.log"}
            if any(artifacts[key]["path"] != str((attempt / relative).resolve())
                   for key, relative in fixed_paths.items()):
                raise ValueError("sampling receipt does not bind expected artifact paths")
            inputs = _prepared_inputs(attempt / "prep/prepared.json", task, attempt)
            if any(artifacts[key] != record for key, record in inputs.items()):
                raise ValueError("sampling receipt does not bind prepared inputs")
            outcome = _sampling_outcome(attempt, task)
            if status != outcome["code"] or receipt.get("sampling") != outcome["sampling"]:
                raise ValueError("sampling receipt/status mismatch")
            return {**base, "status": status, "integrity": "VERIFIED",
                    "qc_status": "FAIL", "pipeline_complete": False,
                    "density_g_cm3": None, "sampling": outcome["sampling"]}
        if status == "FAILED":
            return {**base, "status": "FAILED", "error": receipt.get("error")}
        return {**base, "status": "INCOMPLETE", "error": receipt.get("error")}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {**base, "status": "INCOMPLETE", "error": f"integrity check failed: {exc}"}


class StopRequested(Exception):
    pass


class SamplingStopped(ValueError):
    def __init__(self, outcome):
        super().__init__(outcome["code"])
        self.outcome = outcome


def _sampling_outcome(attempt, task):
    """A special exit code alone is never trusted as a statistical outcome."""
    from thermal_properties.simulation import verify_sampling_outcome
    value = _object(attempt / "density" / "sampling_outcome.json")
    _identity(value, task)
    if (value.get("schema_version") != "polymer-density-sampling-outcome/v1"
            or value.get("smiles") != task["smiles"]
            or value.get("attempt_id") != attempt.name
            or not isinstance(value.get("code"), str)
            or value.get("code") not in SAMPLING_STATUSES
            or value.get("pipeline_complete") is not False
            or value.get("qc_status") != "FAIL"
            or value.get("density_g_cm3") is not None
            or value.get("request_sha256") != _sha(attempt / "request.json")
            or value.get("config_sha256") != _sha(attempt / "density" / "mace_config.json")):
        raise ValueError("invalid sampling outcome identity")
    record = value.get("evidence")
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError("invalid sampling evidence artifact record")
    evidence_path = _contained(Path(record["path"]), attempt)
    if _artifact(evidence_path) != record:
        raise ValueError("sampling evidence hash/size mismatch")
    try:
        evidence = verify_sampling_outcome(evidence_path)
    except (RuntimeError, KeyError, TypeError) as exc:
        raise ValueError(f"sampling evidence failed verification: {exc}") from exc
    if evidence != value.get("sampling") or evidence.get("code") != value["code"]:
        raise ValueError("sampling evidence contents changed")
    return value


def _copy_verified_record(record, destination):
    source = Path(record["path"])
    if _artifact(source) != record:
        raise ValueError(f"parent artifact changed before copy: {source}")
    with source.open("rb") as inp, destination.open("xb") as out:
        shutil.copyfileobj(inp, out)
        out.flush()
        os.fsync(out.fileno())
    copied = _artifact(destination)
    if copied["sha256"] != record["sha256"] or copied["bytes"] != record["bytes"]:
        raise ValueError(f"parent artifact changed during copy: {source}")
    return copied


def _copy_parent_preparation(selector, task, attempt, receipt):
    """Copy only authenticated preparation inputs, never parent trajectories."""
    records = selector["smiles_evidence"]
    evidence_dir = attempt / "prep" / "parent_evidence"
    evidence_dir.mkdir()
    for label in ("request", "receipt", "prepared"):
        receipt["artifacts"]["parent_" + label] = _copy_verified_record(
            records[label], evidence_dir / (label + ".json"))
    old_receipt = _object(evidence_dir / "receipt.json")
    prepared = _object(evidence_dir / "prepared.json")
    _identity(prepared, task)
    for label, filename in (("input_data", "input.data"), ("snapshot_metadata", "input.snapshot.json")):
        destination = attempt / "prep" / filename
        receipt["artifacts"][label] = _copy_verified_record(old_receipt["artifacts"][label], destination)
        prepared[label] = str(destination)
    prepared["parent_preparation_reuse"] = {
        "parent_prepared_sha256": records["prepared"]["sha256"],
        "preparation_processes_started": 0,
        "input_bytes_unchanged": True,
    }
    _atomic_json(attempt / "prep" / "prepared.json", prepared)
    for label in ("stdout", "stderr"):
        log = attempt / ("prepare." + label + ".log")
        with log.open("x", encoding="utf-8") as handle:
            if label == "stdout":
                handle.write("COPIED_VERIFIED_PARENT_PREPARATION; PREPARATION_PROCESSES_STARTED=0\n")
        receipt["artifacts"]["prepare_" + label] = _artifact(log)
    receipt["preparation_action"] = "COPIED_VERIFIED_PARENT_INPUTS"
    _atomic_json(attempt / "receipt.json", receipt)


class StopFlag:
    requested = False


@contextmanager
def _signals(flag):
    originals = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def stop(_signum, _frame):
        flag.requested = True

    for sig in originals:
        signal.signal(sig, stop)
    try:
        yield
    finally:
        for sig, handler in originals.items():
            signal.signal(sig, handler)


def _terminate(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=_ADAPTER_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    # A exited group leader can leave descendants; always close the group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _stage(name, argv, attempt, receipt, flag, claim_descriptor):
    if flag.requested:
        raise StopRequested("interrupted before starting the next stage")
    stdout = attempt / (name + ".stdout.log")
    stderr = attempt / (name + ".stderr.log")
    record = {"stage": name, "argv": argv, "started_at": _now(), "returncode": None}
    receipt["stages"].append(record)
    _atomic_json(attempt / "receipt.json", receipt)
    process = None
    try:
        with stdout.open("xb") as out, stderr.open("xb") as err:
            if flag.requested:
                raise StopRequested("interrupted before child process creation")
            # The adapter retains the task claim even if its dispatcher is
            # killed abruptly; another dispatcher must not duplicate it.
            process = subprocess.Popen(argv, cwd=attempt, stdout=out, stderr=err,
                                       start_new_session=True, shell=False,
                                       pass_fds=(claim_descriptor,))
            record["pid"] = process.pid
            _atomic_json(attempt / "receipt.json", receipt)
            while process.poll() is None:
                if flag.requested:
                    _terminate(process)
                    raise StopRequested("interrupted; child process group terminated")
                time.sleep(0.05)
            if flag.requested:
                _terminate(process)
                raise StopRequested("interrupted after stage completion")
            if process.returncode != 0:
                if name == "density" and process.returncode in {75, 76}:
                    outcome = _sampling_outcome(attempt, receipt)
                    expected_exit = 76 if outcome["code"] == "SAMPLING_QC_FAILED" else 75
                    if process.returncode != expected_exit:
                        raise ValueError("sampling outcome/exit code mismatch")
                    receipt["artifacts"]["sampling_outcome"] = _artifact(attempt / "density" / "sampling_outcome.json")
                    raise SamplingStopped(outcome)
                raise ValueError(f"{name} exited with status {process.returncode}")
    finally:
        if process is not None:
            record["returncode"] = process.returncode
        record["ended_at"] = _now()
        for label, path in (("stdout", stdout), ("stderr", stderr)):
            if path.is_file():
                receipt["artifacts"][name + "_" + label] = _artifact(path)
        _atomic_json(attempt / "receipt.json", receipt)


def _new_attempt(task, task_root, site_path, site, flag, claim_descriptor, parent_selector=None,
                 stage="all", prepared_selector=None):
    previous = _attempts(task_root)
    number = previous[-1][0] + 1 if previous else 1
    attempt = task_root / f"attempt_{number:04d}"
    attempt.mkdir()
    (attempt / "prep").mkdir()
    (attempt / "density").mkdir()
    request_path = attempt / "request.json"
    request = {"schema_version": "polymer-smiles-task/v1", **task,
               "seed": task_seed(task["smiles_sha256"]), "attempt_id": attempt.name,
               "attempt_root": str(attempt), "prepared_manifest_path": str(attempt / "prep" / "prepared.json")}
    if parent_selector is not None:
        request["density_parent_restart"] = parent_selector
    if stage != "all":
        request["execution_stage"] = stage
    if prepared_selector is not None:
        request["prepared_parent"] = prepared_selector
    _atomic_json(request_path, request)
    receipt = {"schema_version": RECEIPT_SCHEMA, **task, "attempt_id": attempt.name,
               "attempt_root": str(attempt), "status": "RUNNING", "started_at": _now(),
               "site_path": str(site_path), "site_sha256": site["_batch_site_sha256"],
               "owner": {"pid": os.getpid(), "hostname": socket.gethostname()},
               "stages": [], "artifacts": {"request": _artifact(request_path)}}
    _atomic_json(attempt / "receipt.json", receipt)
    if stage != "all":
        receipt["execution_stage"] = stage
    try:
        for name, key, directory in (("prepare", "prepare_argv", "prep"), ("density", "density_argv", "density")):
            if stage == "prepare" and name == "density":
                continue
            if _sha(site_path) != receipt["site_sha256"]:
                raise ValueError("site configuration changed during this attempt")
            if name == "prepare" and prepared_selector is not None:
                if flag.requested:
                    raise StopRequested("interrupted before CPU preparation intake")
                _copy_prepared_parent(prepared_selector, task, attempt, receipt)
            elif name == "prepare" and parent_selector is not None:
                if flag.requested:
                    raise StopRequested("interrupted before parent preparation intake")
                _copy_parent_preparation(parent_selector, task, attempt, receipt)
            else:
                command = _argv(site[key], request_path, attempt / directory, site_path)
                _stage(name, command, attempt, receipt, flag, claim_descriptor)
            if _sha(site_path) != receipt["site_sha256"]:
                raise ValueError("site configuration changed during this attempt")
            if name == "prepare":
                prepared = attempt / "prep" / "prepared.json"
                receipt["artifacts"].update(_prepared_inputs(prepared, task, attempt))
                receipt["artifacts"]["prepared"] = _artifact(prepared)
                if stage == "prepare":
                    receipt["artifacts"].update(_preparation_evidence(prepared, task, attempt))
        if stage == "prepare":
            receipt.update(status="PREPARED_QC_PASS", pipeline_complete=False,
                           classical_qc_status="PASS", density_g_cm3=None)
        else:
            result_path = attempt / "density" / "result.json"
            result, manifest = _result(result_path, task, attempt)
            receipt["artifacts"].update(result=_artifact(result_path), run_manifest=_artifact(manifest))
            receipt["status"] = "COMPLETE_QC_PASS" if result["qc_status"] == "PASS" else "COMPLETE_QC_FAIL"
    except SamplingStopped as exc:
        receipt.update(status=exc.outcome["code"], error=str(exc),
                       pipeline_complete=False, qc_status="FAIL",
                       sampling=exc.outcome["sampling"])
    except StopRequested as exc:
        receipt.update(status="INCOMPLETE", error=str(exc))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        receipt.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
    finally:
        receipt["ended_at"] = _now()
        # Even an adapter's failure result is preserved and hashed when present.
        for label, path in (("prepared", attempt / "prep" / "prepared.json"), ("result", attempt / "density" / "result.json")):
            if path.is_file() and not path.is_symlink():
                receipt["artifacts"].setdefault(label, _artifact(path))
        _atomic_json(attempt / "receipt.json", receipt)
        if stage == "prepare":
            (attempt / "receipt.json").chmod(0o444)
    return _summary(task, task_root)


def _run_task(task, work_root, site_path, site, flag, retry_failed, continue_density_from=None,
              stage="all", prepared_from=None):
    task_root = work_root / task["task_id"]
    if task_root.is_symlink():
        raise ValueError(f"task directory cannot be a symlink: {task_root}")
    task_root.mkdir(exist_ok=True)
    lock_path = task_root / ".claim.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"task_id": task["task_id"], "status": "INCOMPLETE", "active_claim": True,
                    "error": "another process owns this task; no duplicate work started"}
        current = _summary(task, task_root)
        if current["status"] in COMPLETE_STATUSES:
            return {**current, "skipped": True}
        if current["status"] == "PREPARED_QC_PASS" and stage != "density":
            return {**current, "skipped": True}
        handoff_ready = stage == "density" and current["status"] == "PREPARED_QC_PASS"
        if current["status"] != "NOT_STARTED" and not handoff_ready and not retry_failed and continue_density_from is None:
            return {**current, "skipped": True, "retry_required": True}
        if flag.requested:
            return {"task_id": task["task_id"], "status": "INCOMPLETE", "error": "interrupted before attempt creation"}
        selector = None
        if continue_density_from is not None:
            from thermal_properties.density_parent import smiles_parent_selector
            selector = smiles_parent_selector(continue_density_from, task)
        prepared_selector = prepared_parent_selector(prepared_from, task) if stage == "density" else None
        return _new_attempt(task, task_root, site_path, site, flag, descriptor, selector,
                            stage, prepared_selector)
    finally:
        os.close(descriptor)


def _work_root(path: Path, *, create: bool):
    if not path.is_absolute():
        raise ValueError("--work-root must be an absolute external path")
    root = path.resolve()
    if root == Path(root.anchor) or root == PACKAGE_ROOT.resolve() or PACKAGE_ROOT.resolve() in root.parents:
        raise ValueError("work-root must be a dedicated directory outside the public package")
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def _parser():
    parser = argparse.ArgumentParser(prog="python -m polymer_batch.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "plan", "run", "status"):
        command = commands.add_parser(name)
        command.add_argument("--task-index", type=int)
        command.add_argument("--start", type=int)
        command.add_argument("--stop", type=int)
        command.add_argument("--shard-index", type=int)
        command.add_argument("--shard-count", type=int)
        if name in {"plan", "run"}:
            command.add_argument("--site", type=Path, required=name == "run")
            command.add_argument("--stage", choices=("all", "prepare", "density"), default="all")
            command.add_argument("--prepared-from", type=Path,
                                 help="verified terminal CPU prepare attempt; required for --stage density")
            command.add_argument("--continue-density-from", type=Path,
                                 help="explicit terminal transition attempt; new identity, no preparation rerun")
        if name in {"run", "status"}:
            command.add_argument("--work-root", type=Path, required=True)
        if name == "run":
            command.add_argument("--confirm-run", choices=("YES",), required=True)
            command.add_argument("--retry-failed", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        tasks = select_tasks(load_catalog(PACKAGE_ROOT / "inputs" / "smiles.csv"),
            **{key: getattr(args, key) for key in ("task_index", "start", "stop", "shard_index", "shard_count")})
        code = 0
        parent_attempt = getattr(args, "continue_density_from", None)
        stage = getattr(args, "stage", "all")
        prepared_from = getattr(args, "prepared_from", None)
        if stage == "density":
            if prepared_from is None or len(tasks) != 1 or not prepared_from.is_absolute():
                raise ValueError("--stage density requires one task and absolute --prepared-from")
        elif prepared_from is not None:
            raise ValueError("--prepared-from is only valid for --stage density")
        if parent_attempt is not None and stage != "all":
            raise ValueError("--continue-density-from cannot be combined with split --stage")
        if prepared_from is not None and args.command == "run":
            destination, parent_root = args.work_root.resolve(), prepared_from.resolve()
            if destination == parent_root or parent_root in destination.parents:
                raise ValueError("new work-root must not be inside immutable CPU parent attempt")
        if parent_attempt is not None:
            if len(tasks) != 1 or not parent_attempt.is_absolute():
                raise ValueError("continue-density-from requires one task and an absolute attempt path")
            if getattr(args, "retry_failed", False):
                raise ValueError("choose continue-density-from or retry-failed, not both")
            if args.command == "run":
                destination = args.work_root.resolve()
                parent_root = parent_attempt.resolve()
                if destination == parent_root or parent_root in destination.parents:
                    raise ValueError("new work-root must not be inside the immutable parent attempt")
        if args.command in {"list", "plan"}:
            result = {"selected_count": len(tasks), "tasks": tasks, "processes_started": 0}
            if args.command == "plan":
                result.update(mode="PLAN_ONLY", execution_order="sequential", retry_automatic=False)
                result["execution_stage"] = stage
                if args.site is not None:
                    site = _site(args.site, stage=stage)
                    names = ("prepare_argv", "density_argv") if stage == "all" else (stage + "_argv",)
                    result["stage_argv_templates"] = {key: site[key] for key in names}
                if prepared_from is not None:
                    result.update(prepared_parent=prepared_parent_selector(prepared_from, tasks[0]),
                                  preparation_action="COPY_VERIFIED_CPU_PARENT_INPUTS")
                if parent_attempt is not None:
                    from thermal_properties.density_parent import smiles_parent_selector
                    result.update(density_parent_restart=smiles_parent_selector(parent_attempt, tasks[0]),
                                  preparation_action="COPY_VERIFIED_PARENT_INPUTS",
                                  initialization_action="SKIP_USE_PARENT_CHECKPOINT")
        elif args.command == "status":
            root = _work_root(args.work_root, create=False)
            summaries = [_summary(task, root / task["task_id"]) for task in tasks]
            result = {"selected_count": len(tasks), "counts": dict(Counter(item["status"] for item in summaries)), "tasks": summaries}
        else:
            site = _site(args.site, stage=stage)
            root = _work_root(args.work_root, create=True)
            summaries = []
            flag = StopFlag()
            with _signals(flag):
                for task in tasks:
                    if flag.requested:
                        break
                    try:
                        summaries.append(_run_task(task, root, args.site.resolve(), site, flag, args.retry_failed,
                                                   parent_attempt, stage, prepared_from))
                    except (OSError, ValueError, KeyError) as exc:
                        summaries.append({"task_id": task["task_id"], "status": "FAILED", "error": str(exc)})
                    print(f"{task['task_id']}: {summaries[-1]['status']}", file=sys.stderr, flush=True)
            successful = {"COMPLETE_QC_PASS", "PREPARED_QC_PASS"} if stage == "prepare" else {"COMPLETE_QC_PASS"}
            code = 130 if flag.requested else int(any(item["status"] not in successful for item in summaries))
            result = {"selected_count": len(tasks), "processed_count": len(summaries), "interrupted": flag.requested,
                      "counts": dict(Counter(item["status"] for item in summaries)), "tasks": summaries}
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
        return code
    except (OSError, ValueError, KeyError) as exc:
        print(f"batch error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
