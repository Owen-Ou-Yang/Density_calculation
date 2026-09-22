"""Offline single-node Slurm/SGE array checks and rendering; never submit jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = ROOT.parent if (ROOT.parent / "LICENSE").is_file() else ROOT
sys.path.insert(0, str(ROOT))
from polymer_batch.catalog import load_catalog, select_tasks
from polymer_batch.cli import _site

FIELDS = {
    "scheduler", "name", "queue", "account", "walltime", "gpu_count",
    "threads_per_rank", "memory_gb", "max_concurrent", "task_start", "task_stop",
    "driver_python", "site_config", "environment_setup", "work_root", "log_dir",
    "sge_pe", "sge_gpu_resource", "sge_memory_resource",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def absolute(value, label, *, kind=None, executable=False, private=False):
    if not isinstance(value, str) or not value or any(c.isspace() or ord(c) < 32 for c in value):
        raise ValueError(f"{label}: use an absolute path without whitespace/control characters")
    path = Path(value)
    if not path.is_absolute() or "REPLACE" in value or "/absolute/path/" in value:
        raise ValueError(f"{label}: configure an actual absolute path")
    # Preserve a venv interpreter's symlink spelling for actual execution.
    resolved = path.resolve()
    if any(c.isspace() or ord(c) < 32 for c in str(resolved)):
        raise ValueError(f"{label}: resolved path contains whitespace/control characters")
    if private and (resolved == PUBLIC_ROOT.resolve() or PUBLIC_ROOT.resolve() in resolved.parents):
        raise ValueError(f"{label}: must be outside the entire public repository")
    if private and resolved == Path(resolved.anchor):
        raise ValueError(f"{label}: use a dedicated private directory")
    if kind == "file" and not path.is_file():
        raise ValueError(f"{label}: file does not exist: {path}")
    if kind == "dir" and not path.is_dir():
        raise ValueError(f"{label}: directory must exist before submission: {path}")
    if executable and not os.access(path, os.X_OK):
        raise ValueError(f"{label}: file must be executable: {path}")
    return path


def integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label}: must be an integer >= {minimum}")
    return value


def word(value, label, pattern=r"[A-Za-z0-9_][A-Za-z0-9_.@:+-]*"):
    if not isinstance(value, str) or not re.fullmatch(pattern, value) or "REPLACE" in value:
        raise ValueError(f"{label}: invalid or placeholder scheduler value")
    return value


def check(profile_path):
    profile_path = absolute(str(profile_path), "profile", kind="file", private=True)
    profile = json.loads(profile_path.read_text())
    if not isinstance(profile, dict) or set(profile) - FIELDS:
        raise ValueError("profile must be an object with only documented fields")
    p = dict(profile)
    if p.get("scheduler") not in {"slurm", "sge"}:
        raise ValueError("scheduler must be slurm or sge")
    for field in ("name", "queue"):
        word(p.get(field), field)
    if p.get("account"):
        word(p["account"], "account")
    else:
        p["account"] = ""
    walltime = p.get("walltime", "")
    if not isinstance(walltime, str) or not re.fullmatch(r"[0-9]+:[0-5][0-9]:[0-5][0-9]", walltime):
        raise ValueError("walltime must be HH:MM:SS")
    if sum(int(v) for v in walltime.split(":")) == 0:
        raise ValueError("walltime must be positive")
    gpus = integer(p.get("gpu_count"), "gpu_count")
    threads = integer(p.get("threads_per_rank"), "threads_per_rank")
    if gpus not in {3, 4} or threads != 4:
        raise ValueError("this release uses 3 or 4 GPUs and 4 threads per rank")
    slots = gpus * threads
    first = integer(p.get("task_start"), "task_start")
    last = integer(p.get("task_stop"), "task_stop")
    tasks = select_tasks(load_catalog(ROOT / "inputs/smiles.csv"), start=first, stop=last)
    concurrency = integer(p.get("max_concurrent"), "max_concurrent")
    if concurrency > len(tasks):
        raise ValueError("max_concurrent exceeds selected task count")
    warnings = [
        "OFFLINE ONLY: MPI/CUDA, environment setup, scheduler policy and physical QC are NOT_TESTED.",
        "One node and one Python coordinator per array element; the site launcher owns MPI.",
        "CPU preparation holds this GPU allocation; benchmark one task before scaling.",
        "Shared POSIX locking and actual memory/storage needs require site validation.",
    ]
    if p["scheduler"] == "slurm":
        integer(p.get("memory_gb"), "memory_gb")
        if any(p.get(k) is not None for k in ("sge_pe", "sge_gpu_resource", "sge_memory_resource")):
            raise ValueError("SGE resource fields must be absent/null for Slurm")
    else:
        if p.get("memory_gb") is not None:
            raise ValueError("SGE memory_gb must be null; use the site's sge_memory_resource")
        p.setdefault("sge_pe", "smp")
        p.setdefault("sge_gpu_resource", "gpu_card")
        word(p["sge_pe"], "sge_pe")
        word(p["sge_gpu_resource"], "sge_gpu_resource", r"[A-Za-z_][A-Za-z0-9_]*")
        memory = p.get("sge_memory_resource")
        if memory is not None:
            word(memory, "sge_memory_resource", r"[A-Za-z_][A-Za-z0-9_]*=[1-9][0-9]*(?:[.][0-9]+)?[kKmMgGtT]?")
            if memory.split("=", 1)[0] in {p["sge_gpu_resource"], "h_rt"}:
                raise ValueError("sge_memory_resource must not override GPU or walltime resources")
        else:
            warnings.append("No SGE memory request: confirm the site's memory resource/limits before submission.")
    records = {}

    def record(label, value, **kwargs):
        path = absolute(str(value), label, kind="file", **kwargs)
        records[label] = {"path": str(path), "sha256": sha(path)}
        return path

    records["profile"] = {"path": str(profile_path), "sha256": sha(profile_path)}
    for field in ("driver_python", "site_config", "environment_setup"):
        path = record(field, p.get(field, ""), executable=field == "driver_python", private=field != "driver_python")
        p[field] = str(path)
    for field in ("work_root", "log_dir"):
        path = absolute(p.get(field), field, kind="dir", private=True)
        if not os.access(path, os.W_OK | os.X_OK):
            raise ValueError(f"{field}: directory must be writable/searchable")
        p[field] = str(path.resolve())
    if p["work_root"] == p["log_dir"]:
        raise ValueError("work_root and log_dir must be separate directories")
    if p["scheduler"] == "slurm" and "%" in p["log_dir"]:
        raise ValueError("Slurm log_dir must not contain filename expansion characters (%)")
    site = _site(Path(p["site_config"]))
    for stage, adapter in (("prepare", "radonpy_prepare.py"), ("density", "mace_density.py")):
        argv = site[stage + "_argv"]
        record(stage + "_python", argv[0], executable=True)
        expected = [argv[0], "-B", str(ROOT / "adapters" / adapter), "--request", "{request}",
                    "--output-dir", "{output_dir}", "--site", "{site}"]
        if argv != expected:
            raise ValueError(f"{stage}_argv: use configure_site.py for this installed package; no outer MPI wrapper")
        record(stage + "_adapter", ROOT / "adapters" / adapter)
    prep = site.get("preparation", {})
    if not isinstance(prep, dict):
        raise ValueError("site.preparation must be an object")
    prep_cpus = integer(prep.get("mpi", 1), "preparation.mpi") * integer(prep.get("omp", 1), "preparation.omp")
    qm_cpus = integer(prep.get("psi4_omp", 1), "preparation.psi4_omp")
    if max(prep_cpus, qm_cpus) > slots:
        raise ValueError("preparation CPU request exceeds allocated CPU slots")
    if prep.get("gpu", 0) != 0:
        raise ValueError("this cluster wrapper keeps classical preparation on CPU")
    qm_memory = integer(prep.get("memory_mb", 1000), "preparation.memory_mb")
    if p["scheduler"] == "slurm" and qm_memory > p["memory_gb"] * 1024:
        raise ValueError("Psi4 memory_mb exceeds Slurm node memory; this is not a peak-memory estimate")
    record("classical_lammps", prep.get("lammps_exec", ""), executable=True)
    density = site.get("density", {})
    if not isinstance(density, dict):
        raise ValueError("site.density must be an object")
    record("model", density.get("model_path", ""))
    if records["model"]["sha256"] != density.get("model_sha256"):
        raise ValueError("checkpoint hash differs from site configuration")
    launcher = record("launcher", density.get("launcher_path", ""), executable=True)
    bundled = ROOT / "launchers/crc_intelmpi_3or4gpu.sh"
    is_crc_launcher = launcher.resolve() == bundled.resolve() or sha(launcher) == sha(bundled)
    if p["scheduler"] == "slurm" and is_crc_launcher:
        raise ValueError("CRC SGE launcher cannot be used under Slurm; configure a qualified site launcher")
    dependencies = density.get("runtime_dependencies", [])
    if not isinstance(dependencies, list):
        raise ValueError("runtime_dependencies must be a list")
    for index, path in enumerate(dependencies):
        record(f"runtime_dependency_{index}", path)
    for label, path in {"catalog": "inputs/smiles.csv", "catalog_identity": "inputs/catalog_identity.json",
                        "protocol": "configs/protocol.json", "density_qc": "thermal_properties/config/density_qc_v1.json",
                        "transition_qc": "thermal_properties/config/mace_transition_qc_v1.json"}.items():
        record(label, ROOT / path)
    return {"schema_version": "cluster-submission-plan/v1", "status": "OFFLINE_CHECK_PASS",
            "profile": p, "selected_count": len(tasks), "mpi_ranks": gpus, "cpu_slots": slots,
            "launcher_kind": "CRC_INTELMPI" if is_crc_launcher else "SITE_PROVIDED_NOT_QUALIFIED_BY_THIS_TOOL",
            "max_simultaneous_gpus": gpus * concurrency, "preparation_cpu_slots": prep_cpus,
            "psi4_cpu_slots": qm_cpus, "records": records, "warnings": warnings,
            "scientific_acceptance": "NOT_TESTED", "scheduler_calls": 0, "model_executions": 0}


def submit_argv(plan, script):
    p = plan["profile"]
    span = f'{p["task_start"]}-{p["task_stop"]}'
    if p["scheduler"] == "slurm":
        argv = ["sbatch", "--job-name=" + p["name"], "--partition=" + p["queue"], "--nodes=1",
                f'--ntasks={p["gpu_count"]}', f'--cpus-per-task={p["threads_per_rank"]}',
                f'--gres=gpu:{p["gpu_count"]}', f'--mem={p["memory_gb"]}G', "--time=" + p["walltime"],
                f'--array={span}%{p["max_concurrent"]}', "--export=ALL", "--no-requeue",
                "--output=" + p["log_dir"] + "/%x-%A_%a.out", "--error=" + p["log_dir"] + "/%x-%A_%a.err"]
        if p["account"]:
            argv += ["--account=" + p["account"]]
    else:
        argv = ["qsub", "-clear", "-S", "/bin/bash", "-q", p["queue"], "-N", p["name"],
                "-pe", p["sge_pe"], str(plan["cpu_slots"]), "-l", f'{p["sge_gpu_resource"]}={p["gpu_count"]}',
                "-l", "h_rt=" + p["walltime"], "-t", span, "-tc", str(p["max_concurrent"]),
                "-r", "n", "-j", "n", "-o", p["log_dir"], "-e", p["log_dir"]]
        if p["account"]:
            argv += ["-P", p["account"]]
        if p.get("sge_memory_resource"):
            argv += ["-l", p["sge_memory_resource"]]
    return argv + [str(script)]


def render_script(plan):
    p = plan["profile"]
    variable = "SLURM_ARRAY_TASK_ID" if p["scheduler"] == "slurm" else "SGE_TASK_ID"
    lines = ["#!/usr/bin/env bash", "# Use submit-command.txt; resource flags are NOT embedded here.",
             "set -euo pipefail", "umask 077", f'task_index="${{{variable}:?scheduler array allocation required}}"',
             '[[ "$task_index" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid array index" >&2; exit 64; }',
             f'(( task_index >= {p["task_start"]} && task_index <= {p["task_stop"]} )) || exit 64']
    if p["scheduler"] == "slurm":
        lines += ['[[ "${SLURM_JOB_NUM_NODES:?}" == 1 ]] || exit 64',
                  f'[[ "${{SLURM_NTASKS:?}}" == {p["gpu_count"]} ]] || exit 64',
                  f'[[ "${{SLURM_GPUS_ON_NODE:?GPU allocation count required}}" == {p["gpu_count"]} ]] || exit 64',
                  f'[[ "${{SLURM_CPUS_PER_TASK:?}}" == {p["threads_per_rank"]} ]] || exit 64']
    else:
        lines += [f'[[ "${{NSLOTS:?}}" == {plan["cpu_slots"]} ]] || exit 64',
                  '[[ -r "${PE_HOSTFILE:?}" ]] || exit 64',
                  "awk '{hosts[$1]=1} END {exit !(length(hosts)==1)}' \"$PE_HOSTFILE\" || exit 64"]
    guard = (
        "import hashlib, pathlib, os\n"
        f"records = {plan['records']!r}\n"
        "for label, record in records.items():\n"
        "    digest = hashlib.sha256()\n"
        "    with pathlib.Path(record['path']).open('rb') as stream:\n"
        "        for chunk in iter(lambda: stream.read(1048576), b''):\n"
        "            digest.update(chunk)\n"
        "    if digest.hexdigest() != record['sha256']:\n"
        "        raise SystemExit('Rendered input changed: ' + label)\n"
    )
    if p["scheduler"] == "sge":
        guard += (
            f"resource = {p['sge_gpu_resource']!r}\n"
            "task = os.environ.get('SGE_HGR_TASK_' + resource, '').strip()\n"
            "host = os.environ.get('SGE_HGR_' + resource, '').strip()\n"
            "tokens = lambda value: value.replace(',', ' ').split()\n"
            "if task and host and tokens(task) != tokens(host):\n"
            "    raise SystemExit('SGE GPU allocation channels disagree')\n"
            "allocated = tokens(task or host)\n"
            f"if len(allocated) != {p['gpu_count']} or len(set(allocated)) != len(allocated):\n"
            "    raise SystemExit('SGE GPU allocation missing, duplicate or wrong count; no preparation started')\n"
        )
    lines += [shlex.join([p["driver_python"], "-B", "-c", guard]),
              "source " + shlex.quote(p["environment_setup"]), "set -euo pipefail"]
    if plan["launcher_kind"] == "CRC_INTELMPI":
        for name in ("CUDA_HOME", "IMPI_MPIRUN", "LAMMPS_RTX6K_IMPI", "IMPI_LIBRARY_PATH", "MACE_CONDA_PREFIX", "THERMAL_PYTHON"):
            lines.append(f': "${{{name}:?required by CRC launcher; configure environment_setup}}"')
        lines += ['[[ -x "$IMPI_MPIRUN" && -x "$LAMMPS_RTX6K_IMPI/lmp" && -x "$THERMAL_PYTHON" ]] || exit 64',
                  '[[ -d "$CUDA_HOME" && -d "$MACE_CONDA_PREFIX" ]] || exit 64']
    lines += [
              f'export MACE_GPU_COUNT={p["gpu_count"]}', f'export MACE_THREADS_PER_RANK={p["threads_per_rank"]}',
              "export PYTHONDONTWRITEBYTECODE=1", "cd " + shlex.quote(str(ROOT)),
              '# Do not wrap this coordinator in srun/mpirun; the density launcher starts MPI.',
              shlex.join(["exec", p["driver_python"], "-B", "-m", "polymer_batch.cli", "run", "--site", p["site_config"],
                          "--work-root", p["work_root"], "--confirm-run", "YES"]) + ' --task-index "$task_index"']
    return "\n".join(lines) + "\n"


def render(plan, output_dir):
    output_dir = absolute(str(output_dir), "output_dir", private=True)
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("output_dir already exists; use a new submission directory")
    output_dir.mkdir(mode=0o700)  # Parent must already exist; no implicit broad directory creation.
    script = output_dir / "job.sh"
    with script.open("x") as handle:
        handle.write(render_script(plan))
    script.chmod(0o700)
    result = dict(plan, script={"path": str(script), "sha256": sha(script)}, submission_argv=submit_argv(plan, script))
    for name, content in (("submission.json", json.dumps(result, indent=2, sort_keys=True) + "\n"),
                          ("submit-command.txt", shlex.join(result["submission_argv"]) + "\n")):
        with (output_dir / name).open("x") as handle:
            handle.write(content)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "render"):
        command = sub.add_parser(name)
        command.add_argument("--profile", type=Path, required=True)
        if name == "render":
            command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = check(args.profile)
        if args.command == "render":
            result = render(result, args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"cluster preflight: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
