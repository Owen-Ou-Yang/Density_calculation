"""Batch boundary tests with real subprocesses and explicitly fake MD output."""

import hashlib
import json
from pathlib import Path
import sys
from unittest import mock

import pytest

from polymer_batch import cli
from test_smiles_parent_intake import make_smiles_parent


PACKAGE = Path(__file__).resolve().parents[1]
FAKE_DENSITY = r'''
import json, pathlib, sys
package, request, output, site_path = sys.argv[1:]
sys.path[:0] = [package, str(pathlib.Path(package) / "tests")]
from adapters import mace_density as adapter
from thermal_properties import simulation as sim
from test_sampling_continuation import synthetic_process, qc_payload
site = json.loads(pathlib.Path(site_path).read_text())
if site["synthetic_mode"].startswith("malformed_"):
    request_data = json.loads(pathlib.Path(request).read_text())
    config = pathlib.Path(output) / "mace_config.json"
    config.write_text("{}")
    outcome = {"schema_version": "polymer-density-sampling-outcome/v1",
        **{key: request_data[key] for key in ("task_id", "smiles", "smiles_sha256", "attempt_id")},
        "code": "SAMPLING_BUDGET_EXHAUSTED", "pipeline_complete": False,
        "qc_status": "FAIL", "density_g_cm3": None,
        "request_sha256": adapter.sha(pathlib.Path(request)), "config_sha256": adapter.sha(config),
        "evidence": None}
    if site["synthetic_mode"] == "malformed_code":
        outcome["code"] = []
    elif site["synthetic_mode"] == "malformed_path":
        outcome["evidence"] = {"path": None}
    (pathlib.Path(output) / "sampling_outcome.json").write_text(json.dumps(outcome))
    raise SystemExit(75)
stages = []
original = sim.execute_thermal_campaign
def evaluate(spec):
    failure = ()
    if spec["stage_role"] == "mace_transition_npt":
        if site["synthetic_mode"] == "exhaust":
            failure = ("minimum_effective_samples",)
        elif site["synthetic_mode"] == "hard":
            failure = ("minimum_effective_samples", "density_drift")
    def qc(_eq, _production):
        value = qc_payload(spec, failure)
        for item in value["policy_results"]:
            item["convergence"]["density"] = {"mean": 1.0, "standard_error": 0.01}
        return value
    return qc
def execute(*args, **kwargs):
    return original(*args, **kwargs, process_runner=synthetic_process(stages),
                    qc_evaluator_factory=evaluate)
sim.execute_thermal_campaign = execute
code = adapter.run(pathlib.Path(request), pathlib.Path(output), pathlib.Path(site_path))
pathlib.Path(site["synthetic_stages"]).write_text(json.dumps(stages))
raise SystemExit(code)
'''


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def bytes_snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def recovery(tmp_path):
    parent = make_smiles_parent(tmp_path / "old")
    launcher = Path(parent["config"]["engine"]["lammps_command"][0])
    launcher.chmod(0o755)
    fake = tmp_path / "fake_density.py"
    fake.write_text(FAKE_DENSITY)
    forbidden_prepare = tmp_path / "prepare_must_not_run.py"
    marker = tmp_path / "unexpected-preparation"
    forbidden_prepare.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise SystemExit(99)\n")
    settings = {"synthetic_mode": "pass", "synthetic_stages": str(tmp_path / "synthetic-stages.json"),
        "prepare_argv": [sys.executable, str(forbidden_prepare), "{request}", "{output_dir}", "{site}"],
        "density_argv": [sys.executable, str(fake), str(PACKAGE), "{request}", "{output_dir}", "{site}"],
        "density": {"model_path": parent["config"]["system"]["mace_model"],
            "model_sha256": cli._sha(Path(parent["config"]["system"]["mace_model"])),
            "launcher_path": str(launcher), "runtime_dependencies": [str(launcher)]}}
    site = tmp_path / "site.json"
    write(site, settings)
    return {**parent, "site": site, "settings": settings, "work": tmp_path / "new", "prepare_marker": marker}


def run_recovery(f, capsys):
    code = cli.main(["run", "--site", str(f["site"]), "--work-root", str(f["work"]),
                     "--task-index", "1", "--continue-density-from", str(f["attempt"]),
                     "--confirm-run", "YES"])
    return code, json.loads(capsys.readouterr().out)


def test_recovery_plan_validates_real_parent_without_starting_process(recovery, capsys):
    f = recovery
    old = bytes_snapshot(f["attempt"])
    with mock.patch("subprocess.Popen", side_effect=AssertionError("plan cannot launch a child")):
        assert cli.main(["plan", "--site", str(f["site"]), "--task-index", "1",
                         "--continue-density-from", str(f["attempt"])]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["processes_started"] == 0
    assert plan["density_parent_restart"]["expected_step"] == 204000
    assert plan["preparation_action"] == "COPY_VERIFIED_PARENT_INPUTS"
    assert not f["work"].exists()
    assert bytes_snapshot(f["attempt"]) == old


@pytest.mark.parametrize("extra", [[], ["--task-index", "2"], ["--task-index", "1", "--retry-failed"]])
def test_invalid_recovery_selection_never_starts_child(recovery, capsys, extra):
    f = recovery
    with mock.patch("subprocess.Popen", side_effect=AssertionError("invalid intake cannot run")):
        assert cli.main(["run", "--site", str(f["site"]), "--work-root", str(f["work"]),
                         "--continue-density-from", str(f["attempt"]), "--confirm-run", "YES", *extra]) != 0
    capsys.readouterr()
    assert not list(f["work"].rglob("attempt_*/request.json"))


def test_tampered_parent_never_starts_child(recovery, capsys):
    f = recovery
    (f["stage"] / "restart.final").write_bytes(b"changed")
    with mock.patch("subprocess.Popen", side_effect=AssertionError("invalid parent cannot run")):
        code, result = run_recovery(f, capsys)
    assert code != 0
    assert not list(f["work"].rglob("attempt_*/request.json"))


@pytest.mark.parametrize("mode,expected,windows", [
    ("pass", "COMPLETE_QC_PASS", 1),
    ("exhaust", "SAMPLING_BUDGET_EXHAUSTED", 2),
    ("hard", "SAMPLING_QC_FAILED", 1),
])
def test_real_subprocess_parent_recovery_receipt_and_status(recovery, capsys, mode, expected, windows):
    f = recovery
    f["settings"]["synthetic_mode"] = mode
    write(f["site"], f["settings"])
    old = bytes_snapshot(f["attempt"])
    code, result = run_recovery(f, capsys)
    row = result["tasks"][0]
    assert row["status"] == expected, row
    assert code == (0 if mode == "pass" else 1)
    assert row["integrity"] == "VERIFIED"
    assert bytes_snapshot(f["attempt"]) == old
    assert not f["prepare_marker"].exists()
    attempt = Path(row["attempt_root"])
    receipt = json.loads((attempt / "receipt.json").read_text())
    assert receipt["preparation_action"] == "COPIED_VERIFIED_PARENT_INPUTS"
    assert [stage["stage"] for stage in receipt["stages"]] == ["density"]
    prepared = json.loads((attempt / "prep/prepared.json").read_text())
    assert prepared["parent_preparation_reuse"]["preparation_processes_started"] == 0
    assert (attempt / "prep/input.data").read_bytes() == Path(f["prepared"]["input_data"]).read_bytes()
    stages = json.loads(Path(f["settings"]["synthetic_stages"]).read_text())
    assert not any(s["stage_role"] == "initialization" for s in stages)
    transitions = [s for s in stages if s["stage_role"] == "mace_transition_npt"]
    assert len(transitions) == windows
    assert transitions[0]["start_step"] == 204000
    assert all(s["production_steps"] == 100000 and not s["initialize_velocities"] for s in transitions)
    if mode == "pass":
        assert row["density_g_cm3"] == 1.0  # Explicitly synthetic, not scientific evidence.
        result_value = json.loads((attempt / "density/result.json").read_text())
        assert result_value["scientific_eligible"] is False
    else:
        assert not (attempt / "density/result.json").exists()
        assert row["density_g_cm3"] is None
        assert row["sampling"]["cumulative_ps"] == (100 if mode == "exhaust" else 75)
        assert not any(s["stage_role"] == "npt_state_point" for s in stages)
        evidence = json.loads((attempt / "density/sampling_outcome.json").read_text())["evidence"]
        Path(evidence["path"]).write_text("{}")
        assert cli._summary(f["task"], attempt.parent)["status"] == "INCOMPLETE"


@pytest.mark.parametrize("exit_code", [75, 76])
def test_special_exit_alone_is_not_a_statistical_receipt(recovery, capsys, exit_code):
    f = recovery
    f["settings"]["density_argv"] = [sys.executable, "-c", f"raise SystemExit({exit_code})",
                                       "{request}", "{output_dir}", "{site}"]
    write(f["site"], f["settings"])
    code, result = run_recovery(f, capsys)
    assert code == 1
    assert result["tasks"][0]["status"] == "FAILED"
    receipt = json.loads((f["work"] / "S000001/attempt_0001/receipt.json").read_text())
    assert receipt["stages"][-1]["returncode"] == exit_code
    assert "sampling_outcome" not in receipt["artifacts"]


@pytest.mark.parametrize("mode", ["malformed_code", "malformed_evidence", "malformed_path"])
def test_malformed_sampling_json_is_a_terminal_failure_not_batch_exception(recovery, capsys, mode):
    f = recovery
    f["settings"]["synthetic_mode"] = mode
    write(f["site"], f["settings"])
    code, result = run_recovery(f, capsys)
    assert code == 1
    assert result["tasks"][0]["status"] == "FAILED"
    receipt = json.loads((f["work"] / "S000001/attempt_0001/receipt.json").read_text())
    assert receipt["status"] == "FAILED" and receipt["ended_at"]
    assert receipt["stages"][-1]["returncode"] == 75
    assert "sampling_outcome" not in receipt["artifacts"]
