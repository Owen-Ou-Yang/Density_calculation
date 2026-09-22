"""Run the existing MACE density PILOT for one newly prepared SMILES task.

No model imports occur until execution. Original SMILES remain an identity
field; the MD engine consumes the collaborator-generated classical structure.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def absolute_file(value, field):
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{field} requires an existing absolute file: {path}")
    return path.resolve()


def request_identity(request):
    if request.get("schema_version") != "polymer-smiles-task/v1":
        raise ValueError("unsupported task schema")
    if hashlib.sha256(request["smiles"].encode("utf-8")).hexdigest() != request["smiles_sha256"]:
        raise ValueError("original SMILES hash mismatch")


def build_config(request, prepared, site, output):
    from thermal_properties.snapshot_contract import load_snapshot_contract
    request_identity(request)
    for key in ("task_id", "smiles_sha256"):
        if prepared.get(key) != request[key]:
            raise ValueError(f"prepared structure belongs to another task: {key}")
    if prepared.get("execution_status") not in (None, "COMPLETE"):
        raise ValueError("preparation did not complete")
    data = absolute_file(prepared["input_data"], "prepared.input_data")
    metadata = absolute_file(prepared["snapshot_metadata"], "prepared.snapshot_metadata")
    contract = load_snapshot_contract(snapshot_path=data, metadata_path=metadata, expected_class="CLASSICAL_EQ2")
    if contract.source_method != "RADONPY_EQ21":
        raise ValueError("the provided preparation adapter must declare its actual RADONPY_EQ21 method")
    elements = prepared["mace_elements"]
    if not isinstance(elements, list) or len(elements) != contract.atom_type_count:
        raise ValueError("mace_elements must contain one element per LAMMPS atom type")
    if set(elements) != set(contract.element_counts):
        raise ValueError("prepared element mapping disagrees with snapshot metadata")
    section, counts, seen_ids = "", {e: 0 for e in elements}, set()
    for line in data.read_text().splitlines():
        bare = line.split("#", 1)[0].strip()
        if not bare:
            continue
        if bare[0].isalpha():
            section = bare
        elif section == "Atoms":
            tokens = bare.split()
            if len(tokens) < 7:
                raise ValueError("prepared data requires atom_style full")
            atom_id, atom_type = int(tokens[0]), int(tokens[2])
            if atom_id in seen_ids or not 1 <= atom_type <= len(elements):
                raise ValueError("duplicate atom ID or invalid atom type")
            if not all(math.isfinite(float(item)) for item in tokens[3:7]):
                raise ValueError("nonfinite prepared atom value")
            seen_ids.add(atom_id)
            counts[elements[atom_type - 1]] += 1
    if counts != dict(contract.element_counts) or len(seen_ids) != contract.atom_count:
        raise ValueError("prepared atom composition/count mismatch")
    settings = site["density"]
    model = absolute_file(settings["model_path"], "density.model_path")
    if sha(model) != settings["model_sha256"]:
        raise ValueError("checkpoint SHA-256 differs from site configuration")
    launcher = absolute_file(settings["launcher_path"], "density.launcher_path")
    if not os.access(launcher, os.X_OK):
        raise ValueError("density launcher is not executable")
    for path in (data, metadata, model, output):
        if any(c.isspace() for c in str(path)):
            raise ValueError("LAMMPS runtime paths must not contain whitespace")
    config = copy.deepcopy(read_json(ROOT / "configs/protocol.json"))
    config["system"].update(polymer_id=request["task_id"], input_data=str(data),
                            snapshot_metadata=str(metadata), snapshot_class="CLASSICAL_EQ2",
                            mace_elements=elements, element_list=list(dict.fromkeys(elements)),
                            mace_model=str(model))
    config["engine"]["lammps_command"] = [str(launcher)]
    config["engine"]["runtime_dependencies"] = [str(absolute_file(x, "runtime_dependency"))
                                                  for x in settings.get("runtime_dependencies", [])]
    config["output_root"] = str(output / "native_runs")
    config["replicas"] = [{"replica_id": "packing_001", "seed": request["seed"]}]
    return config


def run(request_path, output, site_path):
    from thermal_properties.simulation import execute_thermal_campaign, resolve_thermal_config
    from thermal_properties.run_status import build_run_status
    request_path = absolute_file(request_path, "request")
    site_path = absolute_file(site_path, "site")
    output = Path(output)
    if not output.is_absolute():
        raise ValueError("output-dir must be absolute")
    output = output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("runtime output must be outside the public release")
    output.mkdir(parents=True, exist_ok=True)
    request, site = read_json(request_path), read_json(site_path)
    prepared_path = absolute_file(request["prepared_manifest_path"], "prepared manifest")
    prepared = read_json(prepared_path)
    config = build_config(request, prepared, site, output)
    config_path = output / "mace_config.json"
    write_new(config_path, config)
    resolve_thermal_config(config_path)
    run_id = request["task_id"] + "_" + request["attempt_id"]
    manifest_path = execute_thermal_campaign(config_path, run_id=run_id)
    status = build_run_status(manifest_path)
    # Only the final 300 K property branch may supply the density. Never substitute
    # RadonPy preparation density or the 305 K MACE transition value.
    densities = []
    for state in status["state_points"]:
        if state.get("target_temperature_K") == 300.0:
            for policy in state["policy_results"]:
                if policy.get("role") != "density_target":
                    continue
                value = policy.get("density_mean_g_cm3")
                if value is not None:
                    densities.append((value, policy.get("density_standard_error_g_cm3")))
    result = {
        "schema_version": "polymer-density-result/v1", "task_id": request["task_id"],
        "smiles": request["smiles"], "smiles_sha256": request["smiles_sha256"],
        "attempt_id": request["attempt_id"], "request_sha256": sha(request_path),
        "prepared_manifest_sha256": sha(prepared_path), "config_sha256": sha(config_path),
        "checkpoint_sha256": site["density"]["model_sha256"],
        "execution_status": status["execution_status"], "qc_status": status["qc_status"],
        "run_manifest": str(Path(manifest_path).resolve()),
        "run_manifest_sha256": sha(manifest_path), "run_class": "PILOT",
        "scientific_eligible": False, "experimental_reference_included": False,
        "density_g_cm3": densities[0][0] if len(densities) == 1 else None,
        "density_standard_error_g_cm3": densities[0][1] if len(densities) == 1 else None,
        "run_status": status,
    }
    write_new(output / "result.json", result)
    return 0 if result["execution_status"] == "COMPLETE" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args(argv)
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("density adapter interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return run(args.request, args.output_dir, args.site)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        import sys
        print(f"MACE density stage failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
