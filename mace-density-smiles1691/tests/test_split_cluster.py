"""Split scheduler contracts tested with real Bash/Python, never a scheduler or MD."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest

from test_cluster_tools import ROOT, case, cluster, save


FAKE_PREPARE = r'''
import argparse, hashlib, json, pathlib
p = argparse.ArgumentParser()
for name in ('request', 'output-dir', 'site'):
    p.add_argument('--' + name, required=True)
a = p.parse_args()
request = json.loads(pathlib.Path(a.request).read_text())
out = pathlib.Path(a.output_dir)
def record(path):
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
data = out / 'synthetic.data'
data.write_text('SYNTHETIC, NOT MD: ' + request['task_id'])
r = record(data)
metadata = out / 'synthetic.snapshot.json'
metadata.write_text(json.dumps({'synthetic': True, 'snapshot_sha256': r['sha256'], 'snapshot_bytes': r['bytes']}))
stage = {'index': 3, 'status': 'QC_PASS', 'classical_equilibrium_check': True,
         'ended_utc': 'synthetic', 'lammps_returncodes': [0], 'artifacts': [r],
         'final_data': str(data), 'final_data_sha256': r['sha256']}
(out / 'equilibration_stage_eq0003.json').write_text(json.dumps(stage))
prepared = {k: request[k] for k in ('task_id', 'smiles_sha256')}
prepared.update(input_data=str(data), snapshot_metadata=str(metadata), preparation_provenance={
    'original_smiles': request['smiles'], 'smiles_sha256': request['smiles_sha256'],
    'seed': request['seed'], 'classical_equilibrium_check': True, 'equilibration_history': [stage]})
(out / 'prepared.json').write_text(json.dumps(prepared))
'''


def ready_parents(case, indices=(2, 4)):
    """Create successful CPU receipts via actual coordinator/fake adapter subprocesses."""
    parents = case["root"] / "cpu-results"
    adapter = case["root"] / "synthetic-prepare.py"
    adapter.write_text(FAKE_PREPARE)
    site = case["root"] / "synthetic-prepare-site.json"
    site.write_text(json.dumps({"prepare_argv": [sys.executable, "-B", str(adapter),
        "--request", "{request}", "--output-dir", "{output_dir}", "--site", "{site}"]}))
    for index in indices:
        completed = subprocess.run([sys.executable, "-B", "-m", "polymer_batch.cli", "run",
            "--stage", "prepare", "--task-index", str(index), "--site", str(site),
            "--work-root", str(parents), "--confirm-run", "YES"], cwd=ROOT,
            capture_output=True, text=True, timeout=15)
        assert completed.returncode == 0, (completed.stdout, completed.stderr)
        assert json.loads(completed.stdout)["tasks"][0]["status"] == "PREPARED_QC_PASS"
    case["profile"].update(stage="density", prepared_work_root=str(parents))
    save(case)
    return parents


def cpu_profile(case, scheduler="sge"):
    p = case["profile"]
    p.update(stage="prepare", scheduler=scheduler, gpu_count=0, cpu_slots=16,
             queue="cpu", memory_gb=None if scheduler == "sge" else 32)
    p.pop("threads_per_rank")
    if scheduler == "sge":
        p.update(sge_pe="smp", sge_memory_resource="h_vmem=8G")
    # A CPU-only installation has neither a density interpreter nor a checkpoint.
    case["site"].pop("density_argv")
    case["site"].pop("density")
    case["weights"].unlink()
    case["launcher"].unlink()
    save(case)


def runtime(case):
    events = case["root"] / "coordinator.jsonl"
    guard_events = case["root"] / "guard.jsonl"
    driver = case["root"] / "driver"
    driver.write_text(
        "#!" + sys.executable + "\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['-B', '-c']:\n"
        f"    with open({str(guard_events)!r}, 'a') as out:\n"
        "        out.write(json.dumps({'max_arg_bytes': max(len(arg.encode()) for arg in args), "
        "'parent_records': json.loads(args[3]) if len(args) > 3 else None}) + '\\n')\n"
        "    os.execv(sys.executable, [sys.executable, *args])\n"
        "if args[:3] != ['-B', '-m', 'polymer_batch.cli']:\n"
        "    raise SystemExit('Unexpected interpreter invocation')\n"
        f"with open({str(events)!r}, 'a') as out:\n"
        "    out.write(json.dumps({'argv': args, 'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'), "
        "'mace_gpu': os.environ.get('MACE_GPU_COUNT')}) + '\\n')\n"
    )
    driver.chmod(0o755)
    case["profile"]["driver_python"] = str(driver)
    save(case)
    plan = cluster.check(case["profile_path"])
    result = cluster.render(plan, case["root"] / "submission")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SGE_", "SLURM_")) and k not in {"NSLOTS", "PE_HOSTFILE", "BASH_ENV", "ENV"}}
    stage = case["profile"].get("stage", "all")
    index = "1" if stage == "density" else "2"
    if case["profile"]["scheduler"] == "slurm":
        env.update(SLURM_ARRAY_TASK_ID=index, SLURM_JOB_NUM_NODES="1",
                   SLURM_NTASKS="16" if stage == "prepare" else "4",
                   SLURM_CPUS_PER_TASK="1" if stage == "prepare" else "4")
        if stage != "prepare":
            env["SLURM_GPUS_ON_NODE"] = "4"
    else:
        hosts = case["root"] / "hosts"
        hosts.write_text("synthetic-node 16 cpu@synthetic-node <NULL>\n")
        env.update(SGE_TASK_ID=index, NSLOTS="16", PE_HOSTFILE=str(hosts))
        if stage != "prepare":
            env["SGE_HGR_gpu_card"] = "4 5 6 7"
    return result, events, env


def execute(result, env):
    return subprocess.run(["/bin/bash", result["script"]["path"]], env=env,
                          capture_output=True, text=True, timeout=15)


@pytest.mark.parametrize("scheduler", ["sge", "slurm"])
def test_cpu_has_zero_gpu_requests_and_no_density_dependency(case, scheduler):
    cpu_profile(case, scheduler)
    with mock.patch("subprocess.Popen", side_effect=AssertionError("must not submit anything")):
        plan = cluster.check(case["profile_path"])
        result = cluster.render(plan, case["root"] / "offline")
    assert plan["max_simultaneous_gpus"] == 0
    assert plan["scheduler_calls"] == plan["model_executions"] == 0
    assert not {"model", "launcher", "density_adapter", "density_python"} & plan["records"].keys()
    assert "--stage prepare" in Path(result["script"]["path"]).read_text()
    argv = result["submission_argv"]
    assert not any("gpu_card=" in token or "--gres" in token for token in argv)
    if scheduler == "sge":
        assert argv[argv.index("-pe") + 2] == "16"
    else:
        assert "--ntasks=16" in argv and "--cpus-per-task=1" in argv


@pytest.mark.parametrize("scheduler", ["sge", "slurm"])
def test_cpu_script_runs_without_cuda_or_mace_and_hides_inherited_visibility(case, scheduler):
    cpu_profile(case, scheduler)
    result, events, env = runtime(case)
    for key in list(env):
        if key.startswith(("CUDA", "MACE", "IMPI", "THERMAL")):
            del env[key]
    env.update(CUDA_VISIBLE_DEVICES="0,1,2,3", MACE_GPU_COUNT="4")
    completed = execute(result, env)
    assert completed.returncode == 0, completed.stderr
    event = json.loads(events.read_text())
    assert event["cuda"] == "" and event["mace_gpu"] is None
    assert event["argv"][event["argv"].index("--stage") + 1] == "prepare"
    assert "--prepared-from" not in event["argv"]


@pytest.mark.parametrize("scheduler,name,value", [
    ("sge", "SGE_HGR_gpu_card", "0"),
    ("sge", "SGE_HGR_TASK_gpu_card", "4 7"),
    ("slurm", "SLURM_GPUS_ON_NODE", "4"),
    ("slurm", "SLURM_JOB_GPUS", "0"),
    ("slurm", "SLURM_STEP_GPUS", "0"),
    ("slurm", "SLURM_JOB_GPUS", "0,1"),
])
def test_cpu_profile_rejects_unexpected_real_gpu_allocation(case, scheduler, name, value):
    cpu_profile(case, scheduler)
    result, events, env = runtime(case)
    env[name] = value
    completed = execute(result, env)
    assert completed.returncode != 0
    assert "unexpectedly received a GPU" in completed.stderr
    assert not events.exists()


@pytest.mark.parametrize("setting,value", [("gpu_count", 4), ("cpu_slots", 0),
                                          ("cpu_slots", True), ("sge_gpu_resource", "gpu_card"),
                                          ("sge_memory_resource", "gpu_card=4"),
                                          ("sge_memory_resource", "gpu=4")])
def test_cpu_profile_rejects_invalid_or_hidden_gpu_resource(case, setting, value):
    cpu_profile(case)
    case["profile"][setting] = value
    save(case)
    with pytest.raises(ValueError):
        cluster.check(case["profile_path"])


def test_density_requires_verified_ready_parent_and_no_implicit_fallback(case):
    case["profile"].update(stage="density", prepared_work_root=str(case["root"] / "cpu-results"))
    Path(case["profile"]["prepared_work_root"]).mkdir()
    save(case)
    with pytest.raises(ValueError, match="no verified PREPARED_QC_PASS"):
        cluster.check(case["profile_path"])
    assert not (case["root"] / "submission").exists()


@pytest.mark.parametrize("scheduler", ["sge", "slurm"])
def test_density_freezes_only_ready_parents_and_compact_array_mapping(case, scheduler):
    parents = ready_parents(case)
    case["profile"].update(scheduler=scheduler, memory_gb=None if scheduler == "sge" else 32)
    if scheduler == "sge":
        case["profile"].update(sge_pe="smp", sge_gpu_resource="gpu_card")
    # A GPU installation need not have the CPU preparation environment/engine.
    case["site"].pop("preparation")
    case["site"].pop("prepare_argv")
    case["engine"].unlink()
    result, events, env = runtime(case)
    assert [item["task_index"] for item in result["ready_parents"]] == [2, 4]
    assert [item["array_index"] for item in result["ready_parents"]] == [1, 2]
    assert result["excluded_tasks"] == [{"task_id": "S000003", "reason": "NOT_STARTED"}]
    assert not {"prepare_adapter", "prepare_python", "classical_lammps"} & result["records"].keys()
    assert result["selected_count"] == 2 and result["catalog_selected_count"] == 3
    assert result["scheduler_calls"] == result["model_executions"] == 0
    if scheduler == "sge":
        assert result["submission_argv"][result["submission_argv"].index("-t") + 1] == "1-2"
    else:
        assert "--array=1-2%2" in result["submission_argv"]
    variable = "SGE_TASK_ID" if scheduler == "sge" else "SLURM_ARRAY_TASK_ID"
    for array_index, catalog_index in ((1, 2), (2, 4)):
        env[variable] = str(array_index)
        completed = execute(result, env)
        assert completed.returncode == 0, completed.stderr
        event = json.loads(events.read_text().splitlines()[-1])
        args = event["argv"]
        assert args[args.index("--stage") + 1] == "density"
        assert args[args.index("--task-index") + 1] == str(catalog_index)
        assert args[args.index("--prepared-from") + 1] == str(parents / f"S{catalog_index:06d}/attempt_0001")


@pytest.mark.parametrize("status", ["FAILED", "RUNNING", "INCOMPLETE", "COMPLETE_QC_FAIL"])
def test_density_explicitly_excludes_nonready_parent_without_restarting_preparation(case, status):
    parents = ready_parents(case, (2,))
    attempt = parents / "S000003/attempt_0001"
    attempt.mkdir(parents=True)
    (attempt / "receipt.json").write_text(json.dumps({"status": status}))
    plan = cluster.check(case["profile_path"])
    assert plan["selected_count"] == 1 and plan["effective_max_concurrent"] == 1
    assert {"task_id": "S000003", "reason": status} in plan["excluded_tasks"]
    assert "--stage density" in cluster.render_script(plan)
    assert "--stage prepare" not in cluster.render_script(plan)


@pytest.mark.parametrize("filename", ["receipt.json", "prep/prepared.json", "prep/synthetic.data",
                                      "prep/synthetic.snapshot.json", "prepare.stdout.log",
                                      "prep/equilibration_stage_eq0003.json"])
def test_claimed_ready_parent_tamper_blocks_density_render(case, filename):
    parents = ready_parents(case, (2,))
    path = parents / "S000002/attempt_0001" / filename
    path.chmod(0o600)  # Deliberate adversarial mutation of a synthetic fixture.
    if filename == "receipt.json":
        value = json.loads(path.read_text())
        value["smiles"] = "changed exact SMILES"
        path.write_text(json.dumps(value))
    else:
        path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises((ValueError, OSError)):
        cluster.check(case["profile_path"])
    assert not (case["root"] / "submission").exists()


@pytest.mark.parametrize("change", ["input", "receipt", "symlink"])
def test_ready_parent_changes_after_render_stop_before_gpu_coordinator(case, change):
    parents = ready_parents(case, (2,))
    result, events, env = runtime(case)
    attempt = parents / "S000002/attempt_0001"
    path = attempt / ("receipt.json" if change == "receipt" else "prep/synthetic.data")
    path.chmod(0o600)
    if change == "symlink":
        saved = case["root"] / "identical-data"
        path.rename(saved)
        path.symlink_to(saved)
    else:
        path.write_bytes(path.read_bytes() + b" ")
    completed = execute(result, env)
    assert completed.returncode != 0
    assert not events.exists()


def test_later_ready_task_is_not_dynamically_added_to_frozen_gpu_array(case):
    ready_parents(case, (2,))
    result, events, env = runtime(case)
    # Creating another CPU result after render must not alter frozen map or count.
    ready_parents(case, (3,))
    # ready_parents rewrites profile; restore its original frozen bytes for this test.
    profile = result["records"]["profile"]
    # The second helper uses the same profile values, so its bytes remain identical.
    assert cluster.sha(profile["path"]) == profile["sha256"]
    env["SLURM_ARRAY_TASK_ID"] = "2"
    completed = execute(result, env)
    assert completed.returncode != 0
    assert not events.exists()


def test_other_parent_tamper_does_not_abort_this_array_element(case):
    parents = ready_parents(case)
    result, events, env = runtime(case)
    unrelated = parents / "S000004/attempt_0001/prep/synthetic.data"
    unrelated.write_bytes(b"tampered unrelated parent")
    env["SLURM_ARRAY_TASK_ID"] = "1"
    completed = execute(result, env)
    assert completed.returncode == 0, completed.stderr
    assert len(events.read_text().splitlines()) == 1
    guard = json.loads((case["root"] / "guard.jsonl").read_text())
    assert all("S000002" in record["path"] for record in guard["parent_records"].values())
    env["SLURM_ARRAY_TASK_ID"] = "2"
    completed = execute(result, env)
    assert completed.returncode != 0
    assert len(events.read_text().splitlines()) == 1


def test_full_catalog_mapping_keeps_each_process_argument_below_linux_limit(case):
    parents = ready_parents(case, (2,))
    result, events, env = runtime(case)
    real_parent = result["ready_parents"][0]["parent"]
    plan = dict(result)
    plan["records"] = {k: v for k, v in result["records"].items() if not k.startswith("parent_")}
    plan["ready_parents"] = []
    for index in range(1, 1692):
        task_id = f"S{index:06d}"
        if index == 2:
            parent = real_parent
        else:
            # Frozen synthetic unused entries deliberately reference absent files.
            # Successful execution proves they are not read by this array element.
            root = parents / task_id / "attempt_0001"
            artifacts = {label: {**record, "path": str(root / Path(record["path"]).name)}
                         for label, record in real_parent["artifacts"].items()}
            parent = {"attempt_root": str(root), "artifacts": artifacts}
        plan["ready_parents"].append({"array_index": index, "task_index": index,
                                       "task_id": task_id, "parent": parent})
        for label, record in parent["artifacts"].items():
            plan["records"]["parent_" + task_id + "_" + label] = record
    plan.update(scheduler_task_start=1, scheduler_task_stop=1691, selected_count=1691)
    expanded = cluster.render(plan, case["root"] / "full-catalog-render")
    env["SLURM_ARRAY_TASK_ID"] = "2"
    completed = execute(expanded, env)
    assert completed.returncode == 0, completed.stderr
    guard = json.loads((case["root"] / "guard.jsonl").read_text())
    assert guard["max_arg_bytes"] < 128 * 1024
    assert guard["parent_records"] == real_parent["artifacts"]
    invocation = json.loads(events.read_text())
    assert invocation["argv"][invocation["argv"].index("--task-index") + 1] == "2"
    assert expanded["scheduler_calls"] == expanded["model_executions"] == 0


@pytest.mark.parametrize("scheduler", ["sge", "slurm"])
def test_split_gpu_walltime_is_192_hours_without_changing_cpu_budget(case, scheduler):
    gpu = json.loads((ROOT / f"examples/cluster.density.{scheduler}.json").read_text())
    cpu = json.loads((ROOT / f"examples/cluster.prepare.{scheduler}.json").read_text())
    assert gpu["walltime"] == "192:00:00"
    assert cpu["walltime"] == "144:00:00"
    ready_parents(case, (2,))
    case["profile"].update(scheduler=scheduler, walltime=gpu["walltime"])
    if scheduler == "sge":
        case["profile"].update(memory_gb=None, sge_pe="smp", sge_gpu_resource="gpu_card")
    save(case)
    with mock.patch("subprocess.Popen", side_effect=AssertionError("walltime validation must not submit")):
        plan = cluster.check(case["profile_path"])
        result = cluster.render(plan, case["root"] / "walltime-check")
    argv = result["submission_argv"]
    assert ("h_rt=192:00:00" if scheduler == "sge" else "--time=192:00:00") in argv
    assert result["scheduler_calls"] == result["model_executions"] == 0
