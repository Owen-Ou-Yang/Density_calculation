"""Real fake-adapter subprocess tests for CPU/GPU separation; no MD or GPU use."""

import fcntl
import json
from pathlib import Path
import sys

import pytest

from polymer_batch import cli
from polymer_batch.catalog import load_catalog


FAKE_SPLIT_ADAPTER = r'''
import argparse, hashlib, json, pathlib, sys
p=argparse.ArgumentParser()
p.add_argument("stage")
p.add_argument("--request"); p.add_argument("--output-dir"); p.add_argument("--site")
a=p.parse_args(); request=json.loads(pathlib.Path(a.request).read_text())
site=json.loads(pathlib.Path(a.site).read_text()); out=pathlib.Path(a.output_dir)
with open(site["event_log"], "a") as f: f.write(a.stage+"\n")
def write(path, value): path.write_text(json.dumps(value))
def record(path): return {"path": str(path), "bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
identity={k:request[k] for k in ("task_id","smiles_sha256")}
if a.stage == "prepare":
    if site.get("fail_prepare"): raise SystemExit(7)
    data=out/"input.data"; data.write_bytes(b"synthetic CPU prepared atoms\n")
    metadata=out/"input.snapshot.json"; r=record(data)
    write(metadata,{"snapshot_sha256":r["sha256"],"snapshot_bytes":r["bytes"]})
    passed=not site.get("fail_qc", False)
    stage={"index":3,"status":"QC_PASS" if passed else "QC_NOT_CONVERGED",
           "classical_equilibrium_check":passed,"ended_utc":"2026-09-24T00:00:00Z",
           "lammps_returncodes":[0,0,0],"final_data":str(data),
           "final_data_sha256":r["sha256"],"artifacts":[r]}
    write(out/"equilibration_stage_eq0003.json",stage)
    provenance={"original_smiles":request["smiles"],"smiles_sha256":request["smiles_sha256"],
        "seed":request["seed"],"classical_equilibrium_check":passed,"equilibration_history":[stage]}
    if "malformed_history" in site: provenance["equilibration_history"] = site["malformed_history"]
    if "malformed_provenance" in site: provenance = site["malformed_provenance"]
    write(out/"prepared.json",{**identity,"input_data":str(data),"snapshot_metadata":str(metadata),
                              "preparation_provenance":provenance})
else:
    if site.get("fail_density"): raise SystemExit(9)
    prepared=json.loads(pathlib.Path(request["prepared_manifest_path"]).read_text())
    assert pathlib.Path(prepared["input_data"]).read_bytes()==b"synthetic CPU prepared atoms\n"
    manifest=out/"run_manifest.json"; write(manifest,{"synthetic":True})
    if site.get("missing_result"): raise SystemExit(0)
    write(out/"result.json",{**identity,"execution_status":"COMPLETE","qc_status":"PASS",
          "run_manifest":str(manifest),"density_g_cm3":1.0})
'''


def make_split_site(root, *, stage="prepare", **options):
    """Reusable disk-backed fake site; adapter really runs in a subprocess."""
    root.mkdir(parents=True, exist_ok=True)
    script = root / "fake_split_adapter.py"
    if not script.exists():
        script.write_text(FAKE_SPLIT_ADAPTER)
    payload = {"event_log": str(root / "events.log"), **options}
    for name in (("prepare", "density") if stage == "all" else (stage,)):
        payload[name + "_argv"] = [sys.executable, str(script), name, "--request", "{request}",
                                  "--output-dir", "{output_dir}", "--site", "{site}"]
    site = root / (stage + ".site.json")
    site.write_text(json.dumps(payload))
    return site


def make_cpu_parent(root, capsys):
    site = make_split_site(root / "site")
    work = root / "results"
    assert cli.main(["run", "--stage", "prepare", "--site", str(site), "--work-root", str(work),
                     "--task-index", "1", "--confirm-run", "YES"]) == 0
    capsys.readouterr()
    return work / "S000001/attempt_0001"


@pytest.fixture
def split(tmp_path, capsys):
    parent = make_cpu_parent(tmp_path, capsys)
    task = load_catalog(cli.PACKAGE_ROOT / "inputs/smiles.csv")[0]
    return {"root": tmp_path, "parent": parent, "work": parent.parent.parent,
            "task": task, "cpu_site": tmp_path / "site/prepare.site.json"}


def run_density(split, capsys, *, work=None, parent=None, retry=False, **options):
    site = make_split_site(split["root"] / "site", stage="density", **options)
    args = ["run", "--stage", "density", "--prepared-from", str(parent or split["parent"]),
            "--site", str(site), "--work-root", str(work or split["work"]),
            "--task-index", "1", "--confirm-run", "YES"]
    if retry:
        args.append("--retry-failed")
    code = cli.main(args)
    output = capsys.readouterr()
    return code, json.loads(output.out)


def tree_bytes(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_real_cpu_prepare_then_gpu_attempt_preserves_parent_and_identity(split, capsys):
    parent = split["parent"]
    before = tree_bytes(parent)
    receipt = json.loads((parent / "receipt.json").read_text())
    assert receipt["status"] == "PREPARED_QC_PASS"
    assert receipt["pipeline_complete"] is False and receipt["density_g_cm3"] is None
    assert [s["stage"] for s in receipt["stages"]] == ["prepare"]
    assert not (parent / "density.stdout.log").exists()
    assert (parent / "receipt.json").stat().st_mode & 0o222 == 0
    selector = cli.prepared_parent_selector(parent, split["task"])
    assert set(selector["artifacts"]) >= {"receipt", "request", "prepared", "input_data", "snapshot_metadata",
                                          "prepare_stdout", "prepare_stderr", "preparation_stage_eq0003"}
    code, result = run_density(split, capsys)
    assert code == 0 and result["counts"] == {"COMPLETE_QC_PASS": 1}
    child = split["work"] / "S000001/attempt_0002"
    child_receipt = json.loads((child / "receipt.json").read_text())
    assert [s["stage"] for s in child_receipt["stages"]] == ["density"]
    assert child_receipt["preparation_action"] == "COPIED_VERIFIED_CPU_PARENT_INPUTS"
    assert json.loads((child / "request.json").read_text())["smiles"] == split["task"]["smiles"]
    assert (child / "prep/input.data").read_bytes() == (parent / "prep/input.data").read_bytes()
    assert tree_bytes(parent) == before
    assert (split["root"] / "site/events.log").read_text().splitlines() == ["prepare", "density"]


def test_prepare_site_needs_no_density_argv_checkpoint_or_cuda(split, capsys):
    assert "density_argv" not in json.loads(split["cpu_site"].read_text())
    assert cli.main(["plan", "--stage", "prepare", "--site", str(split["cpu_site"]), "--task-index", "1"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert set(plan["stage_argv_templates"]) == {"prepare_argv"}
    assert plan["processes_started"] == 0


def test_completed_prepare_and_density_skip_without_more_children(split, capsys):
    args = ["run", "--stage", "prepare", "--site", str(split["cpu_site"]), "--work-root", str(split["work"]),
            "--task-index", "1", "--confirm-run", "YES"]
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["tasks"][0]["skipped"] is True
    assert run_density(split, capsys)[0] == 0
    assert run_density(split, capsys)[1]["tasks"][0]["skipped"] is True
    assert (split["root"] / "site/events.log").read_text().splitlines() == ["prepare", "density"]


@pytest.mark.parametrize("filename", ["request.json", "prep/input.data", "prep/input.snapshot.json",
    "prep/prepared.json", "prep/equilibration_stage_eq0003.json", "prepare.stdout.log"])
def test_parent_tamper_prevents_density_child(split, capsys, filename):
    path = split["parent"] / filename
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b"\n")
    code, result = run_density(split, capsys, work=split["root"] / "new_gpu_work")
    assert code == 1 and result["counts"] == {"FAILED": 1}
    assert (split["root"] / "site/events.log").read_text().splitlines() == ["prepare"]


def test_failed_classical_qc_cannot_publish_or_launch_density(tmp_path, capsys):
    site = make_split_site(tmp_path / "site", fail_qc=True)
    work = tmp_path / "work"
    assert cli.main(["run", "--stage", "prepare", "--site", str(site), "--work-root", str(work),
                     "--task-index", "1", "--confirm-run", "YES"]) == 1
    assert json.loads(capsys.readouterr().out)["counts"] == {"FAILED": 1}
    task = load_catalog(cli.PACKAGE_ROOT / "inputs/smiles.csv")[0]
    with pytest.raises(ValueError, match="not a terminal"):
        cli.prepared_parent_selector(work / "S000001/attempt_0001", task)
    assert (tmp_path / "site/events.log").read_text().splitlines() == ["prepare"]


@pytest.mark.parametrize("field,value", [("status", "RUNNING"), ("classical_qc_status", "FAIL"),
    ("stages", [None]), ("stages", [{"stage": "prepare", "returncode": 7, "ended_at": "done"}])])
def test_cpu_receipt_claim_tamper_prevents_density_child(split, capsys, field, value):
    path = split["parent"] / "receipt.json"
    path.chmod(0o644)
    receipt = json.loads(path.read_text()); receipt[field] = value
    path.write_text(json.dumps(receipt))
    code, result = run_density(split, capsys, work=split["root"] / "other_gpu_work")
    assert code == 1 and result["counts"] == {"FAILED": 1}
    assert (split["root"] / "site/events.log").read_text().splitlines() == ["prepare"]


def test_receipt_byte_change_after_selector_freeze_is_rejected(split, tmp_path):
    selector = cli.prepared_parent_selector(split["parent"], split["task"])
    path = split["parent"] / "receipt.json"
    path.chmod(0o644); path.write_bytes(path.read_bytes() + b"\n")
    from polymer_batch.preparation_handoff import copy_prepared_parent
    with pytest.raises(ValueError, match="changed after selection"):
        copy_prepared_parent(selector, split["task"], tmp_path / "unused_attempt", {"artifacts": {}})
    assert not (tmp_path / "unused_attempt").exists()


@pytest.mark.parametrize("option,value", [("malformed_provenance", None), ("malformed_provenance", []),
    ("malformed_provenance", "PASS"), ("malformed_history", [None]), ("malformed_history", ["PASS"]),
    ("malformed_history", {}), ("malformed_history", None)])
def test_malformed_cpu_output_creates_failure_receipt_not_running(tmp_path, capsys, option, value):
    site = make_split_site(tmp_path / "site", **{option: value})
    work = tmp_path / "work"
    assert cli.main(["run", "--stage", "prepare", "--site", str(site), "--work-root", str(work),
                     "--task-index", "1", "--confirm-run", "YES"]) == 1
    assert json.loads(capsys.readouterr().out)["counts"] == {"FAILED": 1}
    receipt = json.loads((work / "S000001/attempt_0001/receipt.json").read_text())
    assert receipt["status"] == "FAILED" and receipt["ended_at"]
    assert (tmp_path / "site/events.log").read_text().splitlines() == ["prepare"]


def test_claim_lock_prevents_concurrent_density(split, capsys):
    with (split["parent"].parent / ".claim.lock").open("r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code, result = run_density(split, capsys)
    assert code == 1 and result["tasks"][0]["active_claim"] is True
    assert (split["root"] / "site/events.log").read_text().splitlines() == ["prepare"]


def test_failed_density_needs_explicit_retry_new_attempt(split, capsys):
    assert run_density(split, capsys, fail_density=True)[0] == 1
    assert run_density(split, capsys)[1]["tasks"][0]["retry_required"] is True
    assert (split["root"] / "site/events.log").read_text().splitlines() == ["prepare", "density"]
    assert run_density(split, capsys, retry=True)[0] == 0
    assert (split["work"] / "S000001/attempt_0003/receipt.json").exists()
    assert (split["root"] / "site/events.log").read_text().splitlines() == ["prepare", "density", "density"]


def test_density_exit_zero_without_result_is_not_complete(split, capsys):
    code, result = run_density(split, capsys, missing_result=True)
    assert code == 1 and result["counts"] == {"FAILED": 1}


@pytest.mark.parametrize("extra", [[], ["--prepared-from", "relative"], ["--prepared-from", "/abs", "--continue-density-from", "/old"]])
def test_density_rejects_missing_relative_or_mixed_parent(tmp_path, capsys, extra):
    site = make_split_site(tmp_path / "site", stage="density")
    assert cli.main(["plan", "--stage", "density", "--site", str(site), "--task-index", "1", *extra]) == 2
    assert not (tmp_path / "site/events.log").exists()


def test_cross_task_and_exact_smiles_identity_rejected(split):
    other = load_catalog(cli.PACKAGE_ROOT / "inputs/smiles.csv")[1]
    with pytest.raises(ValueError, match="identity"):
        cli.prepared_parent_selector(split["parent"], other)
    changed = {**split["task"], "smiles": split["task"]["smiles"] + " "}
    with pytest.raises(ValueError, match="terminal"):
        cli.prepared_parent_selector(split["parent"], changed)


@pytest.mark.parametrize("gpu", [1, 4, True, "0", None])
def test_cpu_stage_rejects_gpu_request_before_child(tmp_path, capsys, gpu):
    site = make_split_site(tmp_path / "site", preparation={"gpu": gpu})
    assert cli.main(["run", "--stage", "prepare", "--site", str(site), "--work-root", str(tmp_path / "work"),
                     "--task-index", "1", "--confirm-run", "YES"]) == 2
    assert not (tmp_path / "site/events.log").exists()


def test_prepared_selector_rejects_symlink_parent(split):
    link = split["root"] / "parent_alias"
    link.symlink_to(split["parent"], target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        cli.prepared_parent_selector(link, split["task"])


def test_on_disk_qc_stage_tamper_not_hidden_by_prepared_claim(split):
    path = split["parent"] / "prep/equilibration_stage_eq0003.json"
    data = json.loads(path.read_text()); data["classical_equilibrium_check"] = False
    path.write_text(json.dumps(data))
    receipt_path = split["parent"] / "receipt.json"
    receipt_path.chmod(0o644)
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"]["preparation_stage_eq0003"] = cli._artifact(path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="history differs"):
        cli.prepared_parent_selector(split["parent"], split["task"])
