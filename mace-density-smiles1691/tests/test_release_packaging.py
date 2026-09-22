"""Release transport regressions; no scientific software or scheduler is used."""

import importlib.util
import os
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("public_bundle_check", ROOT / "tools/check_public_bundle.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def test_checked_in_public_package_is_complete():
    result = CHECKER.check(ROOT)
    assert result["status"] == "PASS"
    assert result["smiles"] == 1691
    assert result["experimental_values_included"] is False
    assert result["model_execution"] is False


@pytest.mark.parametrize("script", [
    "launchers/crc_intelmpi_3or4gpu.sh", "examples/crc_array.sh", "examples/slurm_array.sh",
])
def test_shell_script_transport_preserves_executable_mode(script):
    assert os.access(ROOT / script, os.X_OK), f"restore executable Git mode for {script}"


@pytest.mark.parametrize("damage", ["missing_gitignore", "changed_source", "extra_csv", "symlink"])
def test_release_checker_rejects_transport_or_disclosure_damage(tmp_path, damage):
    copy = tmp_path / "release"
    shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    if damage == "missing_gitignore":
        (copy / ".gitignore").unlink()
        expected = "missing files"
    elif damage == "changed_source":
        with (copy / "adapters/mace_density.py").open("a") as handle:
            handle.write("\n# synthetic transport mutation\n")
        expected = "unexpected or changed file"
    elif damage == "extra_csv":
        (copy / "unapproved.csv").write_text("synthetic_column\nnot_real_scientific_data\n")
        expected = "nonpublic artifact"
    else:
        (copy / "unexpected_link").symlink_to(copy / "README.md")
        expected = "symlink"
    with pytest.raises(ValueError, match=expected):
        CHECKER.check(copy)

