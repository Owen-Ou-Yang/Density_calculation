import csv
import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

from adapters import mace_density as adapter
from thermal_properties.snapshot_contract import build_snapshot_contract_payload
from thermal_properties import simulation


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def fixture(tmp_path):
    data = tmp_path / "prep/input.data"
    data.parent.mkdir()
    data.write_text("synthetic test only\n\n4 atoms\n2 atom types\n"
                    "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
                    "Masses\n\n1 12.011\n2 1.008\n\nAtoms # full\n\n"
                    "1 1 1 0 1 1 1\n2 1 2 0 2 1 1\n3 1 2 0 1 2 1\n4 1 2 0 1 1 2\n")
    meta = build_snapshot_contract_payload(snapshot_path=data, snapshot_class="CLASSICAL_EQ2",
                                          packing_id="synthetic", element_counts={"C": 1, "H": 3})
    meta["source_method"] = "RADONPY_EQ21"
    metadata = data.with_suffix(".snapshot.json")
    write(metadata, meta)
    smiles = r"*/C=C\C*"
    request = {"schema_version": "polymer-smiles-task/v1", "task_id": "S000001", "smiles": smiles,
               "smiles_sha256": hashlib.sha256(smiles.encode()).hexdigest(), "seed": 260816,
               "attempt_id": "attempt_0001", "attempt_root": str(tmp_path),
               "prepared_manifest_path": str(data.parent / "prepared.json")}
    prepared = {"task_id": request["task_id"], "smiles_sha256": request["smiles_sha256"],
                "input_data": str(data), "snapshot_metadata": str(metadata), "mace_elements": ["C", "H"]}
    model = tmp_path / "fake-model.pt"
    model.write_bytes(b"not a real model; never loaded")
    launcher = tmp_path / "fake-launcher"
    launcher.write_text("#!/bin/sh\nexit 99\n")
    launcher.chmod(0o755)
    site = {"density": {"model_path": str(model), "model_sha256": adapter.sha(model),
                        "launcher_path": str(launcher), "runtime_dependencies": []}}
    write(tmp_path / "request.json", request)
    write(tmp_path / "site.json", site)
    write(Path(request["prepared_manifest_path"]), prepared)
    return tmp_path, request, prepared, site


def test_configuration_keeps_scientific_protocol_and_exact_original_smiles(fixture):
    root, request, prepared, site = fixture
    config = adapter.build_config(request, prepared, site, root / "density")
    assert config["system"]["polymer_id"] == "S000001"
    assert config["initialize"]["mace_transition_npt_ps"] == 50.0
    assert config["system"]["mace_dtype"] == "float32"
    assert config["system"]["mace_head"] == "omol"
    assert config["npt"]["production_ps"] == 25.0
    assert config["density"]["reference_density_g_cm3"] is None
    assert request["smiles"] == r"*/C=C\C*"


@pytest.mark.parametrize("change", ["smiles", "task", "model", "atoms", "provenance"])
def test_wrong_identity_rejected_before_model_process(fixture, change):
    root, request, prepared, site = fixture
    if change == "smiles":
        request["smiles"] += "C"
    elif change == "task":
        prepared["task_id"] = "S000002"
    elif change == "model":
        Path(site["density"]["model_path"]).write_bytes(b"changed")
    elif change == "atoms":
        Path(prepared["input_data"]).write_text("corrupted")
    else:
        meta = json.loads(Path(prepared["snapshot_metadata"]).read_text())
        meta["source_method"] = "ADEPT_EQ1_EQ2"
        write(Path(prepared["snapshot_metadata"]), meta)
    with mock.patch("subprocess.Popen", side_effect=AssertionError("must not run")):
        with pytest.raises(ValueError):
            adapter.build_config(request, prepared, site, root / "density")


def test_adapter_through_real_execution_core_and_manifest_fake_physics_only(fixture):
    root, request, _, _ = fixture
    stages = []
    original_execute = simulation.execute_thermal_campaign

    def fake_process(invocation):
        stage_dir = invocation.production_segment_path.parents[1]
        spec = json.loads((stage_dir / "stage_spec.json").read_text())
        stages.append(spec)
        eq_end = int(spec["start_step"]) + int(spec["equilibration_steps"])
        end = eq_end + int(spec["production_steps"])
        temp = float(spec.get("state", {}).get("temperature_start_k", 305))
        for path, step in [(invocation.equilibration_segment_path, eq_end), (invocation.production_segment_path, end)]:
            row = {"step": step, "time_ps": step * .00025, "temp_K": temp, "press_bar": 1.01325,
                   "density_g_cm3": 1., "volume_A3": 1000., "pe_eV": -100., "ke_eV": 10.,
                   "etotal_eV": -90., "enthalpy_eV": -89., "pxx_bar": 1.01325, "pyy_bar": 1.01325,
                   "pzz_bar": 1.01325, "pxy_bar": 0., "pxz_bar": 0., "pyz_bar": 0.,
                   "lx_A": 10., "ly_A": 10., "lz_A": 10., "fmax_eV_A": .1, "atom_count": 4}
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
        invocation.stdout_path.write_text("synthetic only")
        invocation.stderr_path.write_text("")
        invocation.log_path.write_text("synthetic only")
        if "state" in spec and int(spec["equilibration_steps"]) > 0:
            (stage_dir / "restart.equilibration").write_bytes(b"synthetic")
        invocation.final_restart_path.write_bytes(b"synthetic")
        return 0

    def execute(*args, **kwargs):
        return original_execute(*args, **kwargs, process_runner=fake_process,
                                qc_evaluator_factory=lambda _spec: lambda _a, _b: "PASS")

    with mock.patch.object(simulation, "execute_thermal_campaign", side_effect=execute), mock.patch(
        "subprocess.Popen", side_effect=AssertionError("no model processes allowed")
    ):
        assert adapter.run(root / "request.json", root / "density", root / "site.json") == 0
    result = json.loads((root / "density/result.json").read_text())
    assert result["smiles"] == request["smiles"]
    assert result["smiles_sha256"] == request["smiles_sha256"]
    assert result["execution_status"] == "COMPLETE"
    assert result["qc_status"] == "PASS"
    assert result["scientific_eligible"] is False
    assert result["run_status"]["manifest_integrity"] == "VERIFIED"
    assert len(stages) == 3
    assert result["run_manifest_sha256"] == adapter.sha(Path(result["run_manifest"]))
