"""Stage-specific site generation must not require or start the other engine."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from polymer_batch import cli


ROOT = Path(__file__).resolve().parents[1]


def configure():
    spec = importlib.util.spec_from_file_location("split_configure_site", ROOT / "tools/configure_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def files(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("configuration must not start any process")
    monkeypatch.setattr("subprocess.Popen", forbidden)
    binary = tmp_path / "engine"
    binary.write_text("fixture only; never executed\n")
    binary.chmod(0o755)
    model = tmp_path / "model"
    model.write_bytes(b"synthetic checkpoint hash fixture")
    return binary, model, tmp_path / "site.json"


def test_cpu_site_requires_no_model_or_mace_environment(files, capsys):
    binary, _, output = files
    configure().main(["--stage", "prepare", "--prep-python", sys.executable,
                      "--classical-lammps", str(binary), "--output", str(output)])
    receipt = json.loads(capsys.readouterr().out)
    site = cli._site(output, stage="prepare")
    assert set(json.loads(output.read_text())) == {"prepare_argv", "preparation"}
    assert site["preparation"]["gpu"] == 0
    assert receipt["model_sha256"] is None
    assert receipt["processes_started"] == 0


def test_gpu_site_requires_no_classical_environment(files, capsys):
    binary, model, output = files
    configure().main(["--stage", "density", "--mace-python", sys.executable,
                      "--model", str(model), "--mace-launcher", str(binary),
                      "--output", str(output)])
    receipt = json.loads(capsys.readouterr().out)
    site = cli._site(output, stage="density")
    assert set(json.loads(output.read_text())) == {"density_argv", "density"}
    assert receipt["model_sha256"] == site["density"]["model_sha256"]
    assert receipt["processes_started"] == 0


@pytest.mark.parametrize("stage", ["all", "prepare", "density"])
def test_selected_stage_dependencies_remain_required(files, stage):
    _, _, output = files
    with pytest.raises(SystemExit) as error:
        configure().main(["--stage", stage, "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_cpu_site_keeps_positive_budget_check(files):
    binary, _, output = files
    with pytest.raises(SystemExit) as error:
        configure().main(["--stage", "prepare", "--prep-python", sys.executable,
                          "--classical-lammps", str(binary), "--prep-mpi", "0",
                          "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_stage_generation_never_overwrites_site(files, capsys):
    binary, _, output = files
    arguments = ["--stage", "prepare", "--prep-python", sys.executable,
                 "--classical-lammps", str(binary), "--output", str(output)]
    configure().main(arguments)
    capsys.readouterr()
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        configure().main(arguments)
    assert output.read_bytes() == before
