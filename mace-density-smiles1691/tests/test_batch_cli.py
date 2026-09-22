"""Real CPU subprocess tests using explicitly synthetic adapters, never MD."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest import mock

import pytest

from polymer_batch import cli
from polymer_batch.catalog import load_catalog, select_tasks, task_seed


FAKE_ADAPTER = r'''
import argparse, json, os, pathlib, signal, subprocess, sys, time
p = argparse.ArgumentParser()
p.add_argument("stage")
p.add_argument("--request", required=True)
p.add_argument("--output-dir", required=True)
p.add_argument("--site", required=True)
a = p.parse_args()
request = json.loads(pathlib.Path(a.request).read_text())
site = json.loads(pathlib.Path(a.site).read_text())
out = pathlib.Path(a.output_dir)
with open(site["test_log"], "a") as log:
    log.write(a.stage + " " + request["task_id"] + "\n")
print(request["smiles"])
print("synthetic " + a.stage + " stderr", file=sys.stderr)
mode = site.get("fake_modes", {}).get(request["task_id"], "success")
identity = {"task_id": request["task_id"], "smiles_sha256": request["smiles_sha256"]}
if a.stage == "prepare":
    if mode == "fail_prepare":
        raise SystemExit(7)
    if mode in {"sleep_prepare", "delayed_nested_cleanup"}:
        child_code = "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); " + "\nfor index in range(3000):\n p.write_text(str(index)); time.sleep(0.02)"
        if mode == "delayed_nested_cleanup":
            child_code = "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); " + child_code
        child = subprocess.Popen([sys.executable, "-c", child_code, site["child_heartbeat"]],
                                 start_new_session=mode == "delayed_nested_cleanup")
        while not pathlib.Path(site["child_heartbeat"]).exists():
            time.sleep(0.01)
        pathlib.Path(site["sleep_marker"]).write_text(json.dumps({"pid": os.getpid(), "child_pid": child.pid}))
        if mode == "delayed_nested_cleanup":
            def interrupted(signum, frame):
                raise KeyboardInterrupt()
            signal.signal(signal.SIGTERM, interrupted)
            try:
                time.sleep(60)
            except KeyboardInterrupt:
                os.killpg(child.pid, signal.SIGTERM)
                time.sleep(4)
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                raise SystemExit(130)
        time.sleep(60)
    if mode != "missing_prepared":
        data = out / "synthetic.data"
        metadata = out / "synthetic.snapshot.json"
        data.write_text("synthetic input; not usable for molecular dynamics\n")
        metadata.write_text(json.dumps({"synthetic": True}))
        (out / "prepared.json").write_text(json.dumps({**identity, "synthetic": True, "input_data": str(data), "snapshot_metadata": str(metadata)}))
else:
    if mode == "missing_result":
        raise SystemExit(0)
    if mode == "wrong_task":
        identity["task_id"] = "S999999"
    manifest = out / "synthetic-run-manifest.json"
    manifest.write_text(json.dumps({**identity, "synthetic": True, "physical_evidence": False}))
    result = {**identity, "execution_status": "COMPLETE", "qc_status": "FAIL" if mode == "qc_fail" else "PASS",
              "run_manifest": str(manifest), "density_g_cm3": 1.0, "density_standard_error_g_cm3": 0.01}
    (out / "result.json").write_text(json.dumps(result))
'''


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def batch_environment(tmp_path, monkeypatch):
    package = tmp_path / "package"
    (package / "inputs").mkdir(parents=True)
    source = package / "inputs" / "smiles.csv"
    raw_smiles = [r"F/C=C\Cl", r"F/C=C/Cl", "N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O"]
    raw_smiles += [f"[{index}C]" for index in range(5, 1692)]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "smiles", "smiles_sha256"])
        for index, smiles in enumerate(raw_smiles, 1):
            writer.writerow([f"S{index:06d}", smiles, hashlib.sha256(smiles.encode()).hexdigest()])
    script = tmp_path / "synthetic_adapter.py"
    script.write_text(FAKE_ADAPTER, encoding="utf-8")
    site = tmp_path / "site.json"
    payload = {"test_log": str(tmp_path / "stage-events.txt"), "sleep_marker": str(tmp_path / "sleep-marker.json"),
               "child_heartbeat": str(tmp_path / "child-heartbeat.txt"), "fake_modes": {}}
    for name in ("prepare", "density"):
        payload[name + "_argv"] = [sys.executable, str(script), name, "--request", "{request}",
                                   "--output-dir", "{output_dir}", "--site", "{site}"]
    write_json(site, payload)
    monkeypatch.setattr(cli, "PACKAGE_ROOT", package)
    return {"package": package, "catalog": source, "site": site, "root": tmp_path / "work", "site_data": payload}


def invoke(environment, capsys, *selection, retry=False):
    argv = ["run", "--site", str(environment["site"]), "--work-root", str(environment["root"]),
            "--confirm-run", "YES", *selection]
    if retry:
        argv.append("--retry-failed")
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, json.loads(captured.out)


def events(environment):
    path = Path(environment["site_data"]["test_log"])
    return path.read_text().splitlines() if path.exists() else []


def test_plan_has_no_process_or_work_directory_and_preserves_raw_strings(batch_environment, capsys):
    with mock.patch.object(cli.subprocess, "Popen", side_effect=AssertionError("plan must not launch anything")):
        assert cli.main(["plan", "--site", str(batch_environment["site"])]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["selected_count"] == 1691
    assert plan["processes_started"] == 0
    assert plan["tasks"][0]["smiles"] == r"F/C=C\Cl"
    assert plan["tasks"][2]["smiles"] != plan["tasks"][3]["smiles"]
    assert not batch_environment["root"].exists()
    assert not events(batch_environment)


def test_selection_shards_are_disjoint_complete_and_ranges_are_inclusive(batch_environment):
    rows = load_catalog(batch_environment["catalog"])
    shards = [select_tasks(rows, shard_index=index, shard_count=7) for index in range(7)]
    flattened = [row["task_id"] for shard in shards for row in shard]
    assert len(flattened) == len(set(flattened)) == 1691
    assert set(flattened) == {row["task_id"] for row in rows}
    assert [row["task_id"] for row in select_tasks(rows, start=3, stop=4)] == ["S000003", "S000004"]
    with pytest.raises(ValueError):
        select_tasks(rows, task_index=1, shard_index=0, shard_count=2)


def test_real_fake_subprocesses_bind_exact_requests_logs_and_status(batch_environment, capsys):
    code, summary = invoke(batch_environment, capsys, "--start", "1", "--stop", "4")
    assert code == 0
    assert summary["counts"] == {"COMPLETE_QC_PASS": 4}
    for row in load_catalog(batch_environment["catalog"])[:4]:
        attempt = batch_environment["root"] / row["task_id"] / "attempt_0001"
        request = json.loads((attempt / "request.json").read_text())
        assert request["smiles"] == row["smiles"]
        assert request["smiles_sha256"] == hashlib.sha256(row["smiles"].encode()).hexdigest()
        assert request["seed"] == task_seed(row["smiles_sha256"])
        assert 1 <= request["seed"] < 900000000
        receipt = json.loads((attempt / "receipt.json").read_text())
        assert set(receipt["artifacts"]) == {"request", "prepared", "input_data", "snapshot_metadata", "result", "run_manifest", "prepare_stdout", "prepare_stderr", "density_stdout", "density_stderr"}
        assert (attempt / "density.stdout.log").read_text().rstrip("\n") == row["smiles"]
    assert cli.main(["status", "--work-root", str(batch_environment["root"]), "--start", "1", "--stop", "4"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["counts"] == {"COMPLETE_QC_PASS": 4}
    assert all(task["integrity"] == "VERIFIED" for task in status["tasks"])
    assert status["tasks"][0]["density_g_cm3"] == 1.0


def test_complete_skip_reverifies_hashes_and_does_not_spawn(batch_environment, capsys):
    assert invoke(batch_environment, capsys, "--task-index", "1")[0] == 0
    before = events(batch_environment)
    code, summary = invoke(batch_environment, capsys, "--task-index", "1")
    assert code == 0 and summary["tasks"][0]["skipped"] is True
    assert events(batch_environment) == before
    result = batch_environment["root"] / "S000001" / "attempt_0001" / "density" / "result.json"
    result.write_text(result.read_text() + "\n")
    code, summary = invoke(batch_environment, capsys, "--task-index", "1")
    assert code == 1 and summary["tasks"][0]["status"] == "INCOMPLETE"
    assert summary["tasks"][0]["retry_required"] is True
    assert events(batch_environment) == before
    code, summary = invoke(batch_environment, capsys, "--task-index", "1", retry=True)
    assert code == 0 and summary["tasks"][0]["attempt_id"] == "attempt_0002"


def test_failure_isolated_and_retry_creates_new_attempt(batch_environment, capsys):
    batch_environment["site_data"]["fake_modes"] = {"S000001": "fail_prepare"}
    write_json(batch_environment["site"], batch_environment["site_data"])
    code, summary = invoke(batch_environment, capsys, "--start", "1", "--stop", "2")
    assert code == 1 and summary["counts"] == {"FAILED": 1, "COMPLETE_QC_PASS": 1}
    assert events(batch_environment) == ["prepare S000001", "prepare S000002", "density S000002"]
    old = batch_environment["root"] / "S000001" / "attempt_0001" / "receipt.json"
    original_receipt = old.read_bytes()
    assert invoke(batch_environment, capsys, "--task-index", "1")[0] == 1
    assert len(events(batch_environment)) == 3
    batch_environment["site_data"]["fake_modes"] = {}
    write_json(batch_environment["site"], batch_environment["site_data"])
    code, summary = invoke(batch_environment, capsys, "--task-index", "1", retry=True)
    assert code == 0 and summary["tasks"][0]["attempt_id"] == "attempt_0002"
    assert old.read_bytes() == original_receipt


@pytest.mark.parametrize("relative", ["request.json", "prep/synthetic.data", "prep/synthetic.snapshot.json"])
def test_skip_rechecks_request_and_prepared_input_bytes(batch_environment, capsys, relative):
    assert invoke(batch_environment, capsys, "--task-index", "1")[0] == 0
    target = batch_environment["root"] / "S000001" / "attempt_0001" / relative
    target.write_bytes(target.read_bytes() + b"\n")
    before = events(batch_environment)
    code, result = invoke(batch_environment, capsys, "--task-index", "1")
    assert code == 1
    assert result["tasks"][0]["status"] == "INCOMPLETE"
    assert result["tasks"][0]["retry_required"] is True
    assert events(batch_environment) == before


@pytest.mark.parametrize("mode", ["missing_prepared", "missing_result", "wrong_task"])
def test_exit_zero_does_not_complete_invalid_artifacts(batch_environment, capsys, mode):
    batch_environment["site_data"]["fake_modes"] = {"S000001": mode}
    write_json(batch_environment["site"], batch_environment["site_data"])
    code, summary = invoke(batch_environment, capsys, "--task-index", "1")
    assert code == 1 and summary["tasks"][0]["status"] == "FAILED"


def test_qc_failure_is_completed_without_physical_pass_or_retry(batch_environment, capsys):
    batch_environment["site_data"]["fake_modes"] = {"S000001": "qc_fail"}
    write_json(batch_environment["site"], batch_environment["site_data"])
    code, summary = invoke(batch_environment, capsys, "--task-index", "1")
    assert code == 1 and summary["tasks"][0]["status"] == "COMPLETE_QC_FAIL"
    before = events(batch_environment)
    code, summary = invoke(batch_environment, capsys, "--task-index", "1", retry=True)
    assert code == 1 and summary["tasks"][0]["skipped"] is True
    assert events(batch_environment) == before


def test_changed_exact_catalog_identity_is_rejected(batch_environment):
    raw = batch_environment["catalog"].read_text()
    batch_environment["catalog"].write_text(raw.replace(r"F/C=C\Cl", r"F/C=C/Cl", 1))
    with pytest.raises(ValueError, match="SHA-256"):
        load_catalog(batch_environment["catalog"])


def test_sigterm_stops_child_group_and_next_task_and_claim_blocks_duplicate(batch_environment, capsys):
    batch_environment["site_data"]["fake_modes"] = {"S000001": "sleep_prepare"}
    write_json(batch_environment["site"], batch_environment["site_data"])
    wrapper = "from pathlib import Path; import sys; from polymer_batch import cli; cli.PACKAGE_ROOT=Path(sys.argv[1]); raise SystemExit(cli.main(sys.argv[2:]))"
    command = [sys.executable, "-c", wrapper, str(batch_environment["package"]), "run", "--site", str(batch_environment["site"]),
               "--work-root", str(batch_environment["root"]), "--confirm-run", "YES", "--start", "1", "--stop", "2"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    marker = Path(batch_environment["site_data"]["sleep_marker"])
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(process.communicate())
            time.sleep(0.02)
        assert marker.exists()
        code, duplicate = invoke(batch_environment, capsys, "--task-index", "1")
        assert code == 1 and duplicate["tasks"][0]["active_claim"] is True
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 130, stderr
        interrupted = json.loads(stdout)
        assert interrupted["interrupted"] is True
        assert interrupted["counts"] == {"INCOMPLETE": 1}
        assert events(batch_environment) == ["prepare S000001"]
        assert not (batch_environment["root"] / "S000002").exists()
        heartbeat = Path(batch_environment["site_data"]["child_heartbeat"])
        last_heartbeat = heartbeat.read_bytes()
        time.sleep(0.15)
        assert heartbeat.read_bytes() == last_heartbeat
        assert invoke(batch_environment, capsys, "--task-index", "1")[0] == 1
        assert events(batch_environment) == ["prepare S000001"]
        batch_environment["site_data"]["fake_modes"] = {}
        write_json(batch_environment["site"], batch_environment["site_data"])
        code, retried = invoke(batch_environment, capsys, "--task-index", "1", retry=True)
        assert code == 0 and retried["tasks"][0]["attempt_id"] == "attempt_0002"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_sigterm_allows_adapter_to_finish_nested_session_cleanup(batch_environment):
    batch_environment["site_data"]["fake_modes"] = {"S000001": "delayed_nested_cleanup"}
    write_json(batch_environment["site"], batch_environment["site_data"])
    wrapper = "from pathlib import Path; import sys; from polymer_batch import cli; cli.PACKAGE_ROOT=Path(sys.argv[1]); raise SystemExit(cli.main(sys.argv[2:]))"
    command = [sys.executable, "-c", wrapper, str(batch_environment["package"]), "run", "--site", str(batch_environment["site"]),
               "--work-root", str(batch_environment["root"]), "--confirm-run", "YES", "--task-index", "1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    marker = Path(batch_environment["site_data"]["sleep_marker"])
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(process.communicate())
            time.sleep(0.02)
        assert marker.exists()
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 130, stderr
        assert json.loads(stdout)["counts"] == {"INCOMPLETE": 1}
        heartbeat = Path(batch_environment["site_data"]["child_heartbeat"])
        last = heartbeat.read_bytes()
        time.sleep(0.15)
        assert heartbeat.read_bytes() == last
        receipt = json.loads((batch_environment["root"] / "S000001/attempt_0001/receipt.json").read_text())
        assert receipt["stages"][0]["returncode"] == 130
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=20)
        if marker.exists():
            child_pid = json.loads(marker.read_text())["child_pid"]
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
