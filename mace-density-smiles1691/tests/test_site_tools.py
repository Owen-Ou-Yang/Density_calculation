"""Interface checks for generated site files and all 1691 catalog selections."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest import mock

from polymer_batch import cli
from polymer_batch.catalog import load_catalog, select_tasks


ROOT = Path(__file__).resolve().parents[1]


def tool(name):
    specification = importlib.util.spec_from_file_location("test_tool_" + name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_configured_site_and_absolute_adapter_paths_work_from_external_cwd(tmp_path, capsys):
    configure = tool("configure_site")
    fake_binary = tmp_path / "fake-engine"
    fake_binary.write_text("synthetic file; never executed\n")
    fake_binary.chmod(0o755)
    model = tmp_path / "synthetic-model.pt"
    model.write_bytes(b"not model weights; SHA-256 fixture only")
    site_path = tmp_path / "site.json"
    with mock.patch("subprocess.Popen", side_effect=AssertionError("site configuration must not execute anything")):
        configure.main([
            "--prep-python", sys.executable, "--mace-python", sys.executable,
            "--classical-lammps", str(fake_binary), "--model", str(model),
            "--mace-launcher", str(fake_binary), "--output", str(site_path),
        ])
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["processes_started"] == 0
    site = cli._site(site_path)
    assert site["density"]["model_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    for name in ("prepare_argv", "density_argv"):
        command = site[name]
        assert Path(command[0]).is_absolute()
        assert command[1] == "-B"
        assert Path(command[2]).is_file() and Path(command[2]).is_absolute()
        # Both actual adapters must start and parse help from the task cwd
        # without importing chemistry/model dependencies or running a stage.
        result = subprocess.run(command[:3] + ["--help"], cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "--request" in result.stdout and "--output-dir" in result.stdout
    with mock.patch("subprocess.Popen", side_effect=AssertionError("plan must not start either adapter")):
        assert cli.main(["plan", "--site", str(site_path)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["selected_count"] == len(plan["tasks"]) == 1691
    assert plan["tasks"][0]["task_id"] == "S000001"
    assert plan["tasks"][-1]["task_id"] == "S001691"


def test_actual_catalog_all_shards_cover_original_order_once():
    rows = load_catalog(ROOT / "inputs/smiles.csv")
    expected = [row["task_id"] for row in rows]
    assert [row["task_id"] for row in select_tasks(rows)] == expected
    for count in (1, 3, 7, 64):
        shards = [select_tasks(rows, shard_index=index, shard_count=count) for index in range(count)]
        flattened = [row["task_id"] for shard in shards for row in shard]
        assert len(flattened) == len(set(flattened)) == 1691
        assert set(flattened) == set(expected)
    assert select_tasks(rows, start=1, stop=1691) == rows


def test_configure_site_preserves_selected_python_symlinks(tmp_path, capsys):
    configure = tool("configure_site")
    prep_python = tmp_path / "prep-environment" / "bin" / "python"
    mace_python = tmp_path / "mace-environment" / "bin" / "python"
    for path in (prep_python, mace_python):
        path.parent.mkdir(parents=True)
        path.symlink_to(Path(sys.executable).resolve())
    engine = tmp_path / "fake-engine"
    engine.write_text("synthetic executable fixture only\n")
    engine.chmod(0o755)
    model = tmp_path / "fake-model.pt"
    model.write_bytes(b"synthetic fixture only")
    output = tmp_path / "site.json"
    with mock.patch("subprocess.Popen", side_effect=AssertionError("configuration cannot run a process")):
        configure.main([
            "--prep-python", str(prep_python), "--mace-python", str(mace_python),
            "--classical-lammps", str(engine), "--model", str(model),
            "--mace-launcher", str(engine), "--output", str(output),
        ])
    capsys.readouterr()
    site = json.loads(output.read_text())
    assert site["prepare_argv"][0] == str(prep_python)
    assert site["density_argv"][0] == str(mace_python)
    assert site["prepare_argv"][0] != str(prep_python.resolve())
    assert site["density_argv"][0] != str(mace_python.resolve())


def test_summary_export_retains_all_unstarted_tasks_without_zero_densities(tmp_path, capsys):
    export = tool("export_summary")
    work = tmp_path / "not-created"
    output = tmp_path / "summary.csv"
    export.main(["--work-root", str(work), "--output", str(output)])
    capsys.readouterr()
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1691
    assert {row["status"] for row in rows} == {"NOT_STARTED"}
    assert {row["density_g_cm3"] for row in rows} == {""}
    assert not work.exists()
