"""Offline scheduler checks and synthetic Bash execution; never submit or run MD."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("cluster_tool_under_test", ROOT / "tools/cluster.py")
cluster = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cluster)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def case(tmp_path):
    """Actual private files using configure_site.py's exact adapter argument order."""
    engine = tmp_path / "fake-classical-lammps"
    launcher = tmp_path / "fake-site-launcher"
    for path in (engine, launcher):
        path.write_text("#!/bin/sh\necho 'fixture must never execute' >&2\nexit 91\n")
        path.chmod(0o755)
    weights = tmp_path / "synthetic-weights.model"
    weights.write_bytes(b"synthetic hash fixture, not a scientific model\n")
    setup = tmp_path / "environment.sh"
    setup.write_text(":\n")
    work = tmp_path / "private-work"
    logs = tmp_path / "private-logs"
    work.mkdir()
    logs.mkdir()
    site_path = tmp_path / "site.json"
    site = {
        stage + "_argv": [sys.executable, "-B", str(ROOT / "adapters" / adapter),
                          "--request", "{request}", "--output-dir", "{output_dir}", "--site", "{site}"]
        for stage, adapter in (("prepare", "radonpy_prepare.py"), ("density", "mace_density.py"))
    }
    site.update({
        "preparation": {"lammps_exec": str(engine), "mpi": 1, "omp": 4, "psi4_omp": 8,
                        "memory_mb": 1024, "gpu": 0},
        "density": {"model_path": str(weights), "model_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                    "launcher_path": str(launcher), "runtime_dependencies": []},
    })
    profile = {
        "scheduler": "slurm", "name": "density-test", "queue": "gpu", "account": "project-test",
        "walltime": "12:00:00", "gpu_count": 4, "threads_per_rank": 4, "memory_gb": 32,
        "max_concurrent": 2, "task_start": 2, "task_stop": 4,
        "driver_python": sys.executable, "site_config": str(site_path), "environment_setup": str(setup),
        "work_root": str(work), "log_dir": str(logs),
    }
    profile_path = tmp_path / "profile.json"
    write_json(site_path, site)
    write_json(profile_path, profile)
    return {"root": tmp_path, "profile_path": profile_path, "profile": profile,
            "site_path": site_path, "site": site, "weights": weights, "engine": engine,
            "launcher": launcher, "setup": setup, "work": work, "logs": logs}


def save(case):
    write_json(case["site_path"], case["site"])
    write_json(case["profile_path"], case["profile"])


def scheduler(case, name):
    case["profile"]["scheduler"] = name
    if name == "sge":
        case["profile"].update(memory_gb=None, sge_pe="smp", sge_gpu_resource="gpu_card",
                               sge_memory_resource="h_vmem=8G")
    save(case)


@pytest.mark.parametrize("name", ["slurm", "sge"])
def test_check_and_render_are_offline_and_record_actual_files(case, name):
    scheduler(case, name)
    dependency = case["root"] / "runtime-dependency.txt"
    dependency.write_text("local runtime dependency hash fixture\n")
    case["site"]["density"]["runtime_dependencies"] = [str(dependency)]
    save(case)
    with mock.patch("subprocess.Popen", side_effect=AssertionError("offline checks may not spawn a process")):
        plan = cluster.check(case["profile_path"])
        result = cluster.render(plan, case["root"] / "submission")
    assert plan["status"] == "OFFLINE_CHECK_PASS"
    assert plan["scheduler_calls"] == plan["model_executions"] == 0
    assert plan["scientific_acceptance"] == "NOT_TESTED"
    assert plan["selected_count"] == 3
    assert plan["mpi_ranks"] == 4 and plan["cpu_slots"] == 16
    assert plan["preparation_cpu_slots"] == 4 and plan["psi4_cpu_slots"] == 8
    assert plan["max_simultaneous_gpus"] == 8
    for record in result["records"].values():
        assert record["sha256"] == hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
    assert result["records"]["runtime_dependency_0"]["path"] == str(dependency)
    output = Path(result["script"]["path"]).parent
    assert json.loads((output / "submission.json").read_text()) == result
    assert shlex.split((output / "submit-command.txt").read_text()) == result["submission_argv"]
    assert result["script"]["sha256"] == hashlib.sha256((output / "job.sh").read_bytes()).hexdigest()
    assert (output / "job.sh").stat().st_mode & 0o777 == 0o700
    assert list(case["work"].iterdir()) == list(case["logs"].iterdir()) == []


def test_changed_checkpoint_fails_before_render(case):
    case["weights"].write_bytes(b"changed checkpoint bytes")
    with pytest.raises(ValueError, match="checkpoint hash"):
        cluster.check(case["profile_path"])


@pytest.mark.parametrize("setting,value", [
    ("mpi", 0), ("mpi", True), ("omp", -1), ("mpi", 5), ("psi4_omp", 17),
    ("memory_mb", 0), ("memory_mb", 32769), ("gpu", 1),
])
def test_preparation_cpu_memory_and_gpu_budgets_fail_closed(case, setting, value):
    case["site"]["preparation"][setting] = value
    save(case)
    with pytest.raises(ValueError):
        cluster.check(case["profile_path"])


@pytest.mark.parametrize("section", ["preparation", "density"])
@pytest.mark.parametrize("value", [None, [], "invalid-section"])
def test_site_sections_must_be_objects(case, section, value):
    case["site"][section] = value
    save(case)
    with pytest.raises(ValueError, match="must be an object"):
        cluster.check(case["profile_path"])


@pytest.mark.parametrize("value", [True, -1, 1.5, "1024"])
def test_preparation_memory_must_be_a_positive_integer(case, value):
    case["site"]["preparation"]["memory_mb"] = value
    save(case)
    with pytest.raises(ValueError, match="preparation.memory_mb"):
        cluster.check(case["profile_path"])


@pytest.mark.parametrize("field,value", [
    ("gpu_count", 2), ("gpu_count", True), ("threads_per_rank", 8), ("memory_gb", 0),
    ("task_start", 0), ("task_start", True), ("task_start", 5), ("task_stop", 1692),
    ("max_concurrent", 0), ("max_concurrent", 4), ("max_concurrent", 1.5),
    ("scheduler", "local"), ("queue", "gpu;touch-owned"), ("name", "job\n--nodes=2"),
    ("account", "$(id)"), ("walltime", "00:00:00"), ("walltime", "12:60:00"),
    ("task_stop", "4;touch-owned"), ("max_concurrent", "2%3"),
])
def test_invalid_profiles_ranges_concurrency_and_scheduler_injection(case, field, value):
    case["profile"][field] = value
    save(case)
    with pytest.raises(ValueError):
        cluster.check(case["profile_path"])


def test_three_gpu_profile_has_twelve_slots(case):
    case["profile"]["gpu_count"] = 3
    save(case)
    plan = cluster.check(case["profile_path"])
    assert plan["cpu_slots"] == 12 and plan["mpi_ranks"] == 3
    assert plan["max_simultaneous_gpus"] == 6


@pytest.mark.parametrize("field", ["profile", "site_config", "environment_setup", "work_root", "log_dir", "output_dir"])
@pytest.mark.parametrize("redirect", [False, True], ids=["direct", "symlink"])
def test_private_paths_cannot_reach_any_part_of_public_repository(case, monkeypatch, field, redirect):
    public = case["root"] / "public-repository"
    sibling = public / "other-public-package"
    sibling.mkdir(parents=True)
    monkeypatch.setattr(cluster, "PUBLIC_ROOT", public)
    target = sibling / field
    if field in {"work_root", "log_dir"}:
        target.mkdir()
    elif field != "output_dir":
        target.write_text("{}\n")
    supplied = target
    if redirect:
        supplied = case["root"] / (field + "-redirect")
        supplied.symlink_to(target)
    if field == "output_dir":
        plan = cluster.check(case["profile_path"])
        with pytest.raises(ValueError, match="entire public repository"):
            cluster.render(plan, supplied)
        assert not target.exists()
        return
    if field == "profile":
        path = supplied
    else:
        case["profile"][field] = str(supplied)
        save(case)
        path = case["profile_path"]
    with pytest.raises(ValueError, match="entire public repository"):
        cluster.check(path)


@pytest.mark.parametrize("field,value", [
    ("work_root", "/"), ("log_dir", "/"), ("work_root", "relative/work"),
    ("driver_python", "relative/python"), ("site_config", "/absolute/path/site.json"),
    ("environment_setup", "/tmp/white space.sh"), ("log_dir", "/tmp/log\nunsafe"),
])
def test_unsafe_or_placeholder_paths_are_rejected(case, field, value):
    case["profile"][field] = value
    save(case)
    with pytest.raises(ValueError):
        cluster.check(case["profile_path"])


def test_work_and_log_directories_must_exist_and_be_distinct(case):
    case["profile"]["work_root"] = str(case["root"] / "missing")
    save(case)
    with pytest.raises(ValueError, match="directory must exist"):
        cluster.check(case["profile_path"])
    case["profile"]["work_root"] = str(case["logs"])
    save(case)
    with pytest.raises(ValueError, match="separate directories"):
        cluster.check(case["profile_path"])


def test_symlink_cannot_hide_whitespace_in_resolved_path(case):
    target = case["root"] / "private directory with spaces"
    target.mkdir()
    redirect = case["root"] / "private-directory-link"
    redirect.symlink_to(target)
    case["profile"]["work_root"] = str(redirect)
    save(case)
    with pytest.raises(ValueError, match="resolved path contains whitespace"):
        cluster.check(case["profile_path"])


def test_slurm_log_directory_cannot_contain_scheduler_filename_expansions(case):
    logs = case["root"] / "logs-%A"
    logs.mkdir()
    case["profile"]["log_dir"] = str(logs)
    save(case)
    with pytest.raises(ValueError, match="filename expansion characters"):
        cluster.check(case["profile_path"])
    assert list(logs.iterdir()) == []


@pytest.mark.parametrize("field", ["driver_python", "prepare_python", "density_python", "classical_lammps", "launcher"])
def test_all_selected_executables_must_be_executable(case, field):
    nonexecutable = case["root"] / "not-executable"
    nonexecutable.write_text("not executable\n")
    nonexecutable.chmod(0o644)
    if field == "driver_python":
        case["profile"][field] = str(nonexecutable)
    elif field in {"prepare_python", "density_python"}:
        case["site"][field.replace("_python", "_argv")][0] = str(nonexecutable)
    elif field == "classical_lammps":
        case["site"]["preparation"]["lammps_exec"] = str(nonexecutable)
    else:
        case["site"]["density"]["launcher_path"] = str(nonexecutable)
    save(case)
    with pytest.raises(ValueError, match="executable"):
        cluster.check(case["profile_path"])


def test_selected_venv_interpreter_spelling_is_preserved(case):
    interpreters = {}
    for label in ("driver", "prepare", "density"):
        path = case["root"] / (label + "-venv") / "bin/python"
        path.parent.mkdir(parents=True)
        path.symlink_to(Path(sys.executable).resolve())
        interpreters[label] = path
    case["profile"]["driver_python"] = str(interpreters["driver"])
    for stage in ("prepare", "density"):
        case["site"][stage + "_argv"][0] = str(interpreters[stage])
    save(case)
    plan = cluster.check(case["profile_path"])
    for label, path in interpreters.items():
        assert plan["records"][label + "_python"]["path"] == str(path)
        assert str(path) != str(path.resolve())
    script = cluster.render_script(plan)
    assert script.count(str(interpreters["driver"])) >= 2
    assert plan["profile"]["driver_python"] == str(interpreters["driver"])


@pytest.mark.parametrize("stage", ["prepare", "density"])
def test_noncanonical_adapter_commands_and_outer_wrappers_are_rejected(case, stage):
    case["site"][stage + "_argv"].insert(1, "--unexpected-wrapper")
    save(case)
    with pytest.raises(ValueError, match="no outer MPI wrapper"):
        cluster.check(case["profile_path"])


@pytest.mark.parametrize("copied", [False, True], ids=["bundled-path", "identical-copy"])
def test_crc_sge_launcher_cannot_be_selected_under_slurm(case, copied):
    launcher = ROOT / "launchers/crc_intelmpi_3or4gpu.sh"
    if copied:
        case["launcher"].write_bytes(launcher.read_bytes())
        launcher = case["launcher"]
    case["site"]["density"]["launcher_path"] = str(launcher)
    save(case)
    with pytest.raises(ValueError, match="CRC SGE launcher cannot be used under Slurm"):
        cluster.check(case["profile_path"])


@pytest.mark.parametrize("field,value", [
    ("memory_gb", 32), ("sge_pe", "smp -pe mpi 99"), ("sge_gpu_resource", "gpu_card=4"),
    ("sge_memory_resource", "h_vmem=8G,other=1"), ("sge_memory_resource", "gpu_card=8"),
    ("sge_memory_resource", "h_rt=12"),
])
def test_sge_resource_fields_reject_conflicts_and_injection(case, field, value):
    scheduler(case, "sge")
    case["profile"][field] = value
    save(case)
    with pytest.raises(ValueError):
        cluster.check(case["profile_path"])


def test_sge_fields_cannot_leak_into_slurm_profile(case):
    case["profile"]["sge_gpu_resource"] = "gpu_card"
    save(case)
    with pytest.raises(ValueError, match="SGE resource fields"):
        cluster.check(case["profile_path"])


@pytest.mark.parametrize("existing", ["directory", "file", "symlink"])
def test_render_never_overwrites_an_existing_output(case, existing):
    output = case["root"] / "submission"
    sentinel = case["root"] / "sentinel.txt"
    sentinel.write_text("preserve this exact file\n")
    if existing == "directory":
        output.mkdir()
        (output / "job.sh").write_bytes(sentinel.read_bytes())
    elif existing == "file":
        output.write_bytes(sentinel.read_bytes())
    else:
        output.symlink_to(sentinel)
    with pytest.raises(ValueError, match="already exists"):
        cluster.render(cluster.check(case["profile_path"]), output)
    assert sentinel.read_text() == "preserve this exact file\n"
    if existing == "directory":
        assert [p.name for p in output.iterdir()] == ["job.sh"]
        assert (output / "job.sh").read_bytes() == sentinel.read_bytes()
    else:
        assert output.read_bytes() == sentinel.read_bytes()
        assert output.is_symlink() == (existing == "symlink")


@pytest.mark.parametrize("name", ["slurm", "sge"])
def test_generated_commands_bind_one_array_task_and_explicit_resources(case, name):
    scheduler(case, name)
    plan = cluster.check(case["profile_path"])
    script_path = case["root"] / "submission/job.sh"
    argv = cluster.submit_argv(plan, script_path)
    script = cluster.render_script(plan)
    commands = [line for line in script.splitlines() if line.startswith("exec ")]
    assert len(commands) == 1
    coordinator = shlex.split(commands[0])
    assert coordinator[:6] == ["exec", sys.executable, "-B", "-m", "polymer_batch.cli", "run"]
    assert coordinator[-2:] == ["--task-index", "$task_index"]
    assert coordinator[coordinator.index("--work-root") + 1] == str(case["work"])
    assert not any(line.lstrip().startswith(("srun ", "mpirun ", "mpiexec ")) for line in script.splitlines())
    assert "export MACE_GPU_COUNT=4" in script and "export MACE_THREADS_PER_RANK=4" in script
    assert argv[-1] == str(script_path)
    if name == "slurm":
        assert argv[0] == "sbatch"
        for flag in ("--nodes=1", "--ntasks=4", "--cpus-per-task=4", "--gres=gpu:4", "--mem=32G",
                     "--array=2-4%2", "--no-requeue", "--account=project-test"):
            assert flag in argv
        assert "--output=" + str(case["logs"]) + "/%x-%A_%a.out" in argv
        assert "--error=" + str(case["logs"]) + "/%x-%A_%a.err" in argv
        assert "${SLURM_ARRAY_TASK_ID:?" in script
    else:
        assert argv[0] == "qsub" and "-clear" in argv
        assert argv[argv.index("-pe") + 1:argv.index("-pe") + 3] == ["smp", "16"]
        assert argv[argv.index("-t") + 1] == "2-4" and argv[argv.index("-tc") + 1] == "2"
        assert "gpu_card=4" in argv and "h_vmem=8G" in argv and "h_rt=12:00:00" in argv
        assert argv[argv.index("-o") + 1] == argv[argv.index("-e") + 1] == str(case["logs"])
        assert argv[argv.index("-r") + 1] == "n"
        assert "${SGE_TASK_ID:?" in script


def rendered_runtime(case, name):
    """Run the hash verifier as real Python, record only the final batch invocation."""
    scheduler(case, name)
    events = case["root"] / "coordinator-invocations.jsonl"
    driver = case["root"] / "fake-driver-python"
    driver.write_text(
        "#!" + sys.executable + "\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['-B', '-c']:\n"
        "    os.execv(sys.executable, [sys.executable, *args])\n"
        "if args[:3] != ['-B', '-m', 'polymer_batch.cli']:\n"
        "    raise SystemExit('Unexpected fixture interpreter invocation')\n"
        "event = {'argv': args, 'cwd': os.getcwd(), 'gpu_count': os.environ.get('MACE_GPU_COUNT'), "
        "'threads': os.environ.get('MACE_THREADS_PER_RANK')}\n"
        f"with open({str(events)!r}, 'a') as stream:\n"
        "    stream.write(json.dumps(event) + '\\n')\n"
    )
    driver.chmod(0o755)
    case["profile"]["driver_python"] = str(driver)
    save(case)
    result = cluster.render(cluster.check(case["profile_path"]), case["root"] / "runtime-submission")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("SLURM_", "SGE_")) and key not in {"NSLOTS", "PE_HOSTFILE", "BASH_ENV", "ENV"}}
    if name == "slurm":
        env.update(SLURM_ARRAY_TASK_ID="2", SLURM_JOB_NUM_NODES="1", SLURM_NTASKS="4",
                   SLURM_CPUS_PER_TASK="4", SLURM_GPUS_ON_NODE="4")
    else:
        hosts = case["root"] / "pe-hostfile"
        hosts.write_text("offline-node 16 all.q@offline-node <NULL>\n")
        env.update(SGE_TASK_ID="2", NSLOTS="16", PE_HOSTFILE=str(hosts), SGE_HGR_gpu_card="0 1 2 3")
    return Path(result["script"]["path"]), events, env


def execute_script(script, env, cwd):
    return subprocess.run(["/bin/bash", str(script)], cwd=cwd, env=env, capture_output=True, text=True, timeout=15)


CRC_ENVIRONMENT_VARIABLES = (
    "CUDA_HOME", "IMPI_MPIRUN", "LAMMPS_RTX6K_IMPI", "IMPI_LIBRARY_PATH", "MACE_CONDA_PREFIX", "THERMAL_PYTHON",
)


def prepare_fake_crc_environment(case, missing=None):
    """Qualify only path/presence guards; all scientific executables remain synthetic."""
    case["launcher"].write_bytes((ROOT / "launchers/crc_intelmpi_3or4gpu.sh").read_bytes())
    locations = {}
    for name in ("cuda", "lammps", "mpi-libraries", "mace-environment"):
        locations[name] = case["root"] / name
        locations[name].mkdir()
    unexpected = case["root"] / "unexpected-scientific-execution"
    fake_executable = "#!/bin/sh\nprintf invoked > " + shlex.quote(str(unexpected)) + "\nexit 91\n"
    executables = [case["root"] / "fake-mpirun", locations["lammps"] / "lmp", case["root"] / "fake-thermal-python"]
    for path in executables:
        path.write_text(fake_executable)
        path.chmod(0o755)
    values = {
        "CUDA_HOME": locations["cuda"], "IMPI_MPIRUN": executables[0],
        "LAMMPS_RTX6K_IMPI": locations["lammps"], "IMPI_LIBRARY_PATH": locations["mpi-libraries"],
        "MACE_CONDA_PREFIX": locations["mace-environment"], "THERMAL_PYTHON": executables[2],
    }
    setup = ["unset " + " ".join(CRC_ENVIRONMENT_VARIABLES)]
    setup.extend("export " + name + "=" + shlex.quote(str(path)) for name, path in values.items() if name != missing)
    case["setup"].write_text("\n".join(setup) + "\n")
    return unexpected


@pytest.mark.parametrize("missing", CRC_ENVIRONMENT_VARIABLES)
def test_real_bash_crc_missing_mandatory_setup_variable_stops_before_coordinator(case, missing):
    unexpected = prepare_fake_crc_environment(case, missing=missing)
    script, events, env = rendered_runtime(case, "sge")
    assert cluster.check(case["profile_path"])["launcher_kind"] == "CRC_INTELMPI"
    run = execute_script(script, env, case["root"])
    assert run.returncode != 0
    assert missing + ": required by CRC launcher" in run.stderr
    assert not events.exists() and not unexpected.exists()


def test_real_bash_crc_complete_fake_setup_allows_exactly_one_coordinator(case):
    unexpected = prepare_fake_crc_environment(case)
    script, events, env = rendered_runtime(case, "sge")
    assert cluster.check(case["profile_path"])["launcher_kind"] == "CRC_INTELMPI"
    run = execute_script(script, env, case["root"])
    assert run.returncode == 0, run.stderr
    invocations = [json.loads(line) for line in events.read_text().splitlines()]
    assert len(invocations) == 1
    assert invocations[0]["argv"][-2:] == ["--task-index", "2"]
    assert not unexpected.exists()
    assert list(case["work"].iterdir()) == list(case["logs"].iterdir()) == []


@pytest.mark.parametrize("name", ["slurm", "sge"])
def test_real_bash_valid_arrays_invoke_one_coordinator_per_selected_task(case, name):
    script, events, env = rendered_runtime(case, name)
    variable = "SLURM_ARRAY_TASK_ID" if name == "slurm" else "SGE_TASK_ID"
    for task_index in (2, 4):
        env[variable] = str(task_index)
        run = execute_script(script, env, case["root"])
        assert run.returncode == 0, run.stderr
        invocations = [json.loads(line) for line in events.read_text().splitlines()]
        assert len(invocations) == (1 if task_index == 2 else 2)
        invocation = invocations[-1]
        assert invocation["argv"] == ["-B", "-m", "polymer_batch.cli", "run", "--site", str(case["site_path"]),
                                      "--work-root", str(case["work"]), "--confirm-run", "YES",
                                      "--task-index", str(task_index)]
        assert invocation["cwd"] == str(ROOT)
        assert invocation["gpu_count"] == "4" and invocation["threads"] == "4"
    assert list(case["work"].iterdir()) == list(case["logs"].iterdir()) == []


@pytest.mark.parametrize("name", ["slurm", "sge"])
@pytest.mark.parametrize("task_index", [None, "0", "1", "5", "02", "2;touch-injected", "2\n3"])
def test_real_bash_invalid_array_ids_never_invoke_coordinator(case, name, task_index):
    script, events, env = rendered_runtime(case, name)
    variable = "SLURM_ARRAY_TASK_ID" if name == "slurm" else "SGE_TASK_ID"
    if task_index is None:
        env.pop(variable)
    else:
        env[variable] = task_index
    run = execute_script(script, env, case["root"])
    assert run.returncode != 0
    assert not events.exists()


@pytest.mark.parametrize("name", ["slurm", "sge"])
@pytest.mark.parametrize("allocation", [None, "wrong-count"])
def test_real_bash_missing_or_wrong_gpu_allocation_fails_before_coordinator(case, name, allocation):
    script, events, env = rendered_runtime(case, name)
    variable = "SLURM_GPUS_ON_NODE" if name == "slurm" else "SGE_HGR_gpu_card"
    if allocation is None:
        env.pop(variable)
    else:
        env[variable] = "3" if name == "slurm" else "0 1 2"
    run = execute_script(script, env, case["root"])
    assert run.returncode != 0
    assert not events.exists()


@pytest.mark.parametrize("scope", ["task", "global"])
def test_real_bash_sge_supports_task_and_global_resource_variables(case, scope):
    script, events, env = rendered_runtime(case, "sge")
    allocation = env.pop("SGE_HGR_gpu_card")
    env["SGE_HGR_TASK_gpu_card" if scope == "task" else "SGE_HGR_gpu_card"] = allocation
    run = execute_script(script, env, case["root"])
    assert run.returncode == 0, run.stderr
    assert len(events.read_text().splitlines()) == 1


def test_real_bash_sge_accepts_matching_comma_and_space_resource_channels(case):
    script, events, env = rendered_runtime(case, "sge")
    env["SGE_HGR_TASK_gpu_card"] = "0,1,2,3"
    run = execute_script(script, env, case["root"])
    assert run.returncode == 0, run.stderr
    assert len(events.read_text().splitlines()) == 1


def test_real_bash_conflicting_sge_resource_channels_fail_before_coordinator(case):
    script, events, env = rendered_runtime(case, "sge")
    env["SGE_HGR_TASK_gpu_card"] = "4 5 6 7"
    run = execute_script(script, env, case["root"])
    assert run.returncode != 0
    assert "SGE GPU allocation channels disagree" in run.stderr
    assert not events.exists()


def test_real_bash_duplicate_sge_gpu_ids_fail_before_coordinator(case):
    script, events, env = rendered_runtime(case, "sge")
    env["SGE_HGR_gpu_card"] = "0 1 1 2"
    run = execute_script(script, env, case["root"])
    assert run.returncode != 0
    assert not events.exists()


@pytest.mark.parametrize("name,field,value", [
    ("slurm", "SLURM_JOB_NUM_NODES", "2"), ("slurm", "SLURM_NTASKS", "3"),
    ("slurm", "SLURM_CPUS_PER_TASK", "8"), ("sge", "NSLOTS", "12"),
])
def test_real_bash_wrong_cpu_or_node_allocation_never_invokes_coordinator(case, name, field, value):
    script, events, env = rendered_runtime(case, name)
    env[field] = value
    run = execute_script(script, env, case["root"])
    assert run.returncode != 0
    assert not events.exists()


def test_real_bash_multinode_sge_allocation_is_rejected(case):
    script, events, env = rendered_runtime(case, "sge")
    Path(env["PE_HOSTFILE"]).write_text("node-a 8 queue <NULL>\nnode-b 8 queue <NULL>\n")
    run = execute_script(script, env, case["root"])
    assert run.returncode != 0
    assert not events.exists()


@pytest.mark.parametrize("name", ["slurm", "sge"])
def test_real_bash_site_changed_after_render_fails_hash_check_before_invocation(case, name):
    script, events, env = rendered_runtime(case, name)
    case["site"]["preparation"]["omp"] = 2
    write_json(case["site_path"], case["site"])
    run = execute_script(script, env, case["root"])
    assert run.returncode != 0
    assert "Rendered input changed: site_config" in run.stderr
    assert not events.exists()
