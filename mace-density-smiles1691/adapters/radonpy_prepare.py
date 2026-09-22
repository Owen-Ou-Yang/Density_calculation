#!/usr/bin/env python3
"""Prepare one NEW polymer with the official RadonPy 0.2.11 APIs.

This adapter runs real chemistry only when explicitly invoked. Package tests
inject a fake backend. The sequence is RESP monomer/terminal preparation,
GAFF2_mod polymer construction, and EQ21step (packing + 21-step compression /
decompression + NPT sampling). Unconverged classical QC triggers bounded
Additional NPT sampling with unchanged QC (5M-step chunks, 50M-step default
cap); execution/analysis/structure errors never trigger extensions. It does
not read an experimental density.

API sources, audited against the immutable v0.2.11 release:
https://github.com/RadonPy/RadonPy/blob/v0.2.11/radonpy/core/poly.py
https://github.com/RadonPy/RadonPy/blob/v0.2.11/radonpy/core/utils.py
https://github.com/RadonPy/RadonPy/blob/v0.2.11/radonpy/sim/qm.py
https://github.com/RadonPy/RadonPy/blob/v0.2.11/radonpy/sim/preset/eq.py
https://github.com/RadonPy/RadonPy/blob/v0.2.11/radonpy/sim/lammps.py

Prerequisites: RadonPy 0.2.11, RDKit, NumPy/SciPy/pandas, MDTraj, Psi4,
RESP, dftd3, and a classical LAMMPS binary with the packages required by
RadonPy (including molecular force fields, KSPACE, SHAKE, and XTC output).
MPI execution additionally requires the launcher configured for RadonPy.

Unspecified stereochemistry uses the declared RadonPy E/S monomer defaults
and the configured tacticity. Explicitly chiral source units use isotactic
copying (RadonPy's no-inversion mode). These are modeling assumptions, not a
match to an experimental microstructure. Original strings remain unchanged;
the builder's derived representation and effective settings are recorded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from numbers import Real
import os
from pathlib import Path
import random
import re
import sys
from types import SimpleNamespace


RADONPY_VERSION = "0.2.11"
METHOD = "RadonPy.GAFF2_mod.RESP.EQ21step"
DEFAULTS = {
    "target_atoms_per_chain": 600, "chains": 6, "initial_density": 0.05,
    "temperature_k": 300.0, "pressure_atm": 1.0,
    "packing_density": 0.8, "max_temperature_k": 600.0,
    "max_pressure_atm": 50000.0, "time_step_fs": 1.0, "eq_step": 5.0,
    "max_eq_step": 50.0,
    "tacticity": "atactic", "omp": 1, "mpi": 1, "gpu": 0,
    "psi4_omp": 1, "memory_mb": 1000, "nconf": 1000, "dft_nconf": 4,
}
REQUEST_FIELDS = {
    "task_id", "smiles", "smiles_sha256", "seed", "attempt_id", "attempt_root",
}


class PreparationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _absolute(value: object, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not Path(value).is_absolute():
        raise PreparationError(f"{label} must be an absolute path")
    path = Path(value)
    if path.is_symlink():
        raise PreparationError(f"{label} must not be a symlink")
    return path.resolve()


def _write_json_new(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        handle.write("\n")


def _validate(request: dict, output_dir: Path, site: dict) -> tuple[Path, dict]:
    if not isinstance(request, dict) or not REQUEST_FIELDS.issubset(request):
        raise PreparationError("request must contain " + ", ".join(sorted(REQUEST_FIELDS)))
    if request.get("schema_version", "polymer-smiles-task/v1") != "polymer-smiles-task/v1":
        raise PreparationError("unsupported request schema_version")
    if not isinstance(request["task_id"], str) or not re.fullmatch(r"S[0-9]{6}", request["task_id"]):
        raise PreparationError("invalid task_id")
    if not isinstance(request["attempt_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", request["attempt_id"]):
        raise PreparationError("invalid attempt_id")
    smiles = request["smiles"]
    if not isinstance(smiles, str) or not smiles:
        raise PreparationError("smiles must be a nonempty exact source string")
    expected = hashlib.sha256(smiles.encode("utf-8")).hexdigest()
    if request["smiles_sha256"] != expected:
        raise PreparationError("original SMILES hash mismatch")
    seed = request["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2147483647:
        raise PreparationError("seed must be an integer in [0, 2147483647]")
    root = _absolute(request["attempt_root"], "attempt_root")
    output = _absolute(output_dir, "output_dir")
    if output == root or not output.is_relative_to(root):
        raise PreparationError("output_dir must be a child of attempt_root")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise PreparationError("output_dir must be new or empty; no resume or overwrite")
    if not isinstance(site, dict) or not isinstance(site.get("preparation"), dict):
        raise PreparationError("site.preparation must be an object")
    supplied = site["preparation"]
    unknown = set(supplied) - set(DEFAULTS) - {"lammps_exec"}
    if unknown:
        raise PreparationError("unknown preparation parameters: " + ", ".join(sorted(unknown)))
    params = {**DEFAULTS, **supplied}
    binary = _absolute(params.get("lammps_exec"), "preparation.lammps_exec")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise PreparationError("preparation.lammps_exec is not an executable file")
    params["lammps_exec"] = str(binary)
    integer_fields = {"target_atoms_per_chain", "chains", "omp", "mpi", "gpu",
                      "psi4_omp", "memory_mb", "nconf", "dft_nconf"}
    for key in integer_fields:
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "gpu" else 1):
            raise PreparationError(f"invalid positive integer preparation.{key}")
    for key in set(DEFAULTS) - integer_fields - {"tacticity"}:
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise PreparationError(f"invalid positive number preparation.{key}")
    if params["tacticity"] not in {"atactic", "isotactic", "syndiotactic"}:
        raise PreparationError("unsupported tacticity")
    if params["dft_nconf"] > params["nconf"]:
        raise PreparationError("dft_nconf cannot exceed nconf")
    if params["eq_step"] < 5:
        raise PreparationError("eq_step must be >= 5 (millions of NPT sampling steps)")
    _sampling_budget(params)
    return output, params


def _sampling_budget(params: dict) -> tuple[int, int]:
    """Exact whole chunks avoid silently shortening RadonPy's analysis window."""
    values = []
    for key in ("eq_step", "max_eq_step"):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 5:
            raise PreparationError(f"{key} must be finite and >= 5 million steps")
        steps = value * 1_000_000
        if isinstance(steps, float) and not steps.is_integer():
            raise PreparationError(f"{key} must specify a whole number of sampling steps")
        values.append(int(steps))
    chunk, cap = values
    if cap < chunk or cap % chunk:
        raise PreparationError("max_eq_step must be >= eq_step and an exact multiple of eq_step")
    return chunk, cap


def load_backend(params: dict) -> SimpleNamespace:
    """Import external tools at runtime, never during module import or export."""
    os.environ["LAMMPS_EXEC"] = params["lammps_exec"]
    try:
        version = importlib.metadata.version("radonpy-pypi")
        if version != RADONPY_VERSION:
            raise PreparationError(f"RadonPy {RADONPY_VERSION} required; found {version}")
        import numpy as np
        import psi4
        import mdtraj
        import rdkit
        from rdkit import Chem
        from radonpy.core import poly, utils
        from radonpy.ff.gaff2_mod import GAFF2_mod
        from radonpy.sim import lammps, qm
        from radonpy.sim.preset import eq
    except ImportError as exc:
        raise PreparationError(f"missing RadonPy preparation dependency: {exc}") from exc
    versions = {"radonpy-pypi": version, "numpy": np.__version__,
                "rdkit": rdkit.__version__, "psi4": psi4.__version__,
                "mdtraj": mdtraj.__version__}
    for distribution in ("resp", "dftd3", "scipy", "pandas"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise PreparationError(f"required dependency missing: {distribution}") from exc
    return SimpleNamespace(np=np, Chem=Chem, poly=poly, utils=utils,
                           GAFF2_mod=GAFF2_mod, qm=qm, lammps=lammps, eq=eq, versions=versions)


def _require_mol(mol, stage: str):
    if mol is None or not hasattr(mol, "GetNumAtoms") or mol.GetNumAtoms() <= 0:
        raise PreparationError(f"RadonPy failed at {stage}")
    return mol


def _snapshot_metadata(path: Path, mol, packing_id: str) -> tuple[dict, list[str], dict]:
    """Bind final data bytes; atom IDs are RDKit indices+1 in RadonPy's writer."""
    section = None
    masses, atoms, bounds = {}, {}, {}
    n_atoms = n_types = None
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            text = raw.split("#", 1)[0].strip()
            if not text:
                continue
            fields = text.split()
            if re.fullmatch(r"[0-9]+ atoms", text):
                n_atoms = int(fields[0])
            elif re.fullmatch(r"[0-9]+ atom types", text):
                n_types = int(fields[0])
            elif len(fields) == 4 and fields[-2:] in (["xlo", "xhi"], ["ylo", "yhi"], ["zlo", "zhi"]):
                bounds[fields[2][0]] = [float(fields[0]), float(fields[1])]
            elif text[0].isalpha():
                section = text
            elif section == "Masses":
                index, mass = int(fields[0]), float(fields[1])
                if index in masses or not math.isfinite(mass) or mass <= 0:
                    raise PreparationError("invalid/duplicate LAMMPS mass")
                masses[index] = mass
            elif section == "Atoms":
                if len(fields) not in (7, 10):
                    raise PreparationError("final LAMMPS Atoms must use full style")
                index, atom_type = int(fields[0]), int(fields[2])
                if index in atoms or not all(math.isfinite(float(x)) for x in fields[3:7]):
                    raise PreparationError("invalid/duplicate final atom record")
                atoms[index] = atom_type
    if n_atoms != mol.GetNumAtoms() or len(atoms) != n_atoms or set(atoms) != set(range(1, n_atoms + 1)):
        raise PreparationError("final file atom IDs/count differ from returned RadonPy cell")
    if n_types is None or set(masses) != set(range(1, n_types + 1)) or set(atoms.values()) != set(masses):
        raise PreparationError("invalid final atom-type/mass mapping")
    if set(bounds) != {"x", "y", "z"} or not all(math.isfinite(x) for b in bounds.values() for x in b) or not all(b[1] > b[0] for b in bounds.values()):
        raise PreparationError("final data lacks a finite positive orthogonal cell")
    type_elements = {}
    counts = Counter()
    for index, atom_type in atoms.items():
        atom = mol.GetAtomWithIdx(index - 1)
        element = atom.GetSymbol()
        if element == "*" or atom.GetIsotope() != 0:
            raise PreparationError("unresolved linkers/isotopes in final polymer")
        if atom_type in type_elements and type_elements[atom_type] != element:
            raise PreparationError("one LAMMPS type maps to multiple elements")
        if abs(masses[atom_type] - atom.GetMass()) > 0.02:
            raise PreparationError("LAMMPS mass differs from returned atom identity")
        type_elements[atom_type] = element
        counts[element] += 1
    metadata = {
        "schema_version": "thermal-properties-snapshot-contract/v1",
        "snapshot_class": "CLASSICAL_EQ2", "source_method": "RADONPY_EQ21",
        "packing_id": packing_id, "snapshot_sha256": _sha256(path),
        "snapshot_bytes": path.stat().st_size, "atom_count": n_atoms,
        "atom_type_count": n_types, "element_counts": dict(sorted(counts.items())),
    }
    return metadata, [type_elements[i] for i in range(1, n_types + 1)], {
        str(i): {"element": type_elements[i], "mass": masses[i]} for i in range(1, n_types + 1)
    }


def _stage_artifacts(work: Path) -> list[dict]:
    artifacts = []
    for path in sorted(work.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(work.resolve()):
            raise PreparationError("equilibration artifact must not be a symlink or escape stage directory")
        if path.is_file():
            artifacts.append({"path": str(path.resolve()), "bytes": path.stat().st_size,
                              "sha256": _sha256(path)})
    return artifacts


def _finite_values(value, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_values(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_values(item, f"{label}[{index}]")
    elif isinstance(value, Real) and not math.isfinite(value):
        raise PreparationError(f"nonfinite classical analysis value: {label}")


def _analysis_evidence(analysis, properties: dict) -> dict:
    # RadonPy comparisons against NaN do not reliably reject it. Reject
    # nonfinite raw thermo/derived properties before asking its unchanged QC.
    _finite_values(properties, "properties")
    for index, frame in enumerate(getattr(analysis, "dfs", [])):
        _finite_values(frame.to_numpy().tolist(), f"thermo[{index}]")
    checks = {}
    mapping = {"rg_sd_crit": ("rg_data", ("mean_mean", "sd_max")),
               "compress_sma_sd_crit": ("compress_T_data", ("mean", "sma_sd")),
               "volexp_sma_sd_crit": ("volume_exp_data", ("mean", "sma_sd"))}
    for name, criterion in vars(analysis).items():
        if not name.endswith("_crit"):
            continue
        data_name, fields = mapping.get(name, (name.replace("_sma_sd_crit", "_data"), ("mean", "sma_sd")))
        row = {"criterion": criterion, "active": criterion is not None}
        if criterion is not None:
            data = getattr(analysis, data_name, {})
            row["values"] = {field: data.get(field) for field in fields}
            if any(not isinstance(value, Real) or not math.isfinite(value)
                   for value in row["values"].values()):
                raise PreparationError(f"missing/nonfinite classical QC data: {data_name}")
            row["values"] = {field: float(value) for field, value in row["values"].items()}
            row["criterion"] = float(criterion)
        checks[name] = row
    return checks


def _exec_with_exit_audit(api, execute, returncodes: list[int], expected: int):
    engine_class = api.lammps.LAMMPS
    original_exec = engine_class.exec
    def audited_exec(engine, *args, **kwargs):
        completed = original_exec(engine, *args, **kwargs)
        if kwargs.get("return_cmd", False):
            return completed
        code = getattr(completed, "returncode", None)
        if isinstance(code, bool) or not isinstance(code, int):
            raise PreparationError("RadonPy LAMMPS exec did not return an exit code")
        returncodes.append(code)
        if code != 0:
            raise PreparationError(f"equilibration LAMMPS exited with status {code}")
        return completed
    engine_class.exec = audited_exec
    try:
        cell = _require_mol(execute(), "equilibration")
    finally:
        engine_class.exec = original_exec
    if returncodes != [0] * expected:
        raise PreparationError(f"equilibration requires {expected} successful LAMMPS exits; found {returncodes}")
    return cell


def equilibrate_bounded(api, cell, params: dict, work: Path, output: Path, *,
                        initial_preset=None, initial_returncodes=None,
                        completed_steps: int = 0, start_index: int = 4,
                        parent_history=None) -> dict:
    """QC, then bounded Additional NPT chunks using the returned current cell.

    Fresh preparation passes an already executed EQ21 preset and its three
    audited exit codes. A separately verified legacy continuation may omit it,
    pass its saved cell, completed_steps and parent_history; this helper never
    repeats RESP, construction, packing or the compression/decompression stage.
    Neither path changes RadonPy 0.2.11 QC thresholds. Its sampling() fixes the
    NPT step at 1 fs: time_step_fs affects EQ21 packing/compression only.
    """
    chunk, cap = _sampling_budget(params)
    if isinstance(completed_steps, bool) or not isinstance(completed_steps, int):
        raise PreparationError("completed_steps must be an integer")
    if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 4:
        raise PreparationError("continuation start_index must be >= 4")
    work, output = Path(work).resolve(), Path(output).resolve()
    history = list(parent_history or [])
    atom_identity = [(atom.GetSymbol(), atom.GetIsotope(), atom.GetMass()) for atom in cell.GetAtoms()]
    if initial_preset is not None:
        if completed_steps != 0 or initial_returncodes != [0, 0, 0]:
            raise PreparationError("initial EQ21 requires zero prior steps and three audited zero exits")
        completed_steps = chunk
        preset, stage_work, index = initial_preset, work, 3
        codes = list(initial_returncodes)
    else:
        if completed_steps < 5_000_000 or completed_steps >= cap or completed_steps % chunk:
            raise PreparationError("continuation completed_steps must be a whole completed chunk >= 5M and below cap")
        preset, index = None, start_index
    while True:
        kind = "EQ21step" if preset is initial_preset and initial_preset is not None else "Additional"
        record = {"index": index, "kind": kind, "status": "RUNNING",
                  "started_utc": datetime.now(timezone.utc).isoformat(),
                  "sampling_time_step_fs": 1.0, "chunk_sampling_steps": chunk,
                  "max_sampling_steps": cap, "classical_equilibrium_check": None,
                  "lammps_returncodes": [], "artifacts": []}
        record_path = output / f"equilibration_stage_eq{index:04d}.json"
        started_path = output / f"equilibration_stage_eq{index:04d}_started.json"
        if record_path.exists() or started_path.exists():
            raise PreparationError("equilibration stage identity already exists; no overwrite")
        _write_json_new(started_path, record)
        try:
            if kind == "Additional":
                stage_work = work / f"continuation_eq{index:04d}"
                stage_work.mkdir()
                codes = []
                record["lammps_returncodes"] = codes
                preset = api.eq.Additional(cell, idx=index, work_dir=str(stage_work),
                                           solver_path=params["lammps_exec"])
                cell = _exec_with_exit_audit(api, lambda: preset.exec(
                    temp=params["temperature_k"], press=params["pressure_atm"],
                    eq_step=chunk / 1_000_000, omp=params["omp"], mpi=params["mpi"],
                    gpu=params["gpu"]), codes, 1)
                if [(atom.GetSymbol(), atom.GetIsotope(), atom.GetMass()) for atom in cell.GetAtoms()] != atom_identity:
                    raise PreparationError("atom identity changed during additional equilibration")
                completed_steps += chunk
            record["lammps_returncodes"] = codes
            record["completed_sampling_steps"] = completed_steps
            record["work_dir"] = str(stage_work)
            final = stage_work / preset.last_data
            if final.is_symlink() or not final.resolve().is_relative_to(stage_work) or not final.is_file():
                raise PreparationError("RadonPy final data file missing or outside work directory")
            final = final.resolve()
            # A damaged structure is an execution failure, never a reason to
            # add more MD. Validate before QC and before any next chunk.
            _snapshot_metadata(final, cell, "equilibration-stage-validation")
            analysis = preset.analyze()
            properties = analysis.get_all_prop(temp=params["temperature_k"],
                press=params["pressure_atm"], save=True, save_name="analyze")
            record["qc_evidence"] = _analysis_evidence(analysis, properties)
            passed = bool(analysis.check_eq())
            record["classical_equilibrium_check"] = passed
            record["status"] = "QC_PASS" if passed else "QC_NOT_CONVERGED"
            record["final_data"] = str(final)
            record["final_data_sha256"] = _sha256(final)
        except Exception as exc:
            record["status"] = "FAILED"
            record["error"] = str(exc)
            record["exception_type"] = type(exc).__name__
            raise
        finally:
            record["ended_utc"] = datetime.now(timezone.utc).isoformat()
            try:
                if "stage_work" in locals() and stage_work.exists():
                    record["artifacts"] = _stage_artifacts(stage_work)
            except Exception as artifact_exc:
                record["status"] = "FAILED"
                record["classical_equilibrium_check"] = None
                record["artifact_error"] = str(artifact_exc)
                raise
            finally:
                _write_json_new(record_path, record)
                history.append(record)
        if passed:
            return {"cell": cell, "preset": preset, "work": stage_work,
                    "history": history, "completed_steps": completed_steps,
                    "final_path": final}
        if completed_steps >= cap:
            raise PreparationError("RadonPy classical equilibrium check failed at maximum sampling budget; no prepared structure published")
        index = start_index if kind == "EQ21step" else index + 1
        preset = None


def publish_prepared(request: dict, output: Path, params: dict, api,
                     equilibration: dict, build_provenance: dict) -> dict:
    """Publish the verified final classical structure, never a MACE density."""
    if not equilibration["history"] or equilibration["history"][-1].get("classical_equilibrium_check") is not True:
        raise PreparationError("cannot publish without final classical QC PASS")
    final = equilibration["final_path"]
    metadata, elements, mass_mapping = _snapshot_metadata(
        final, equilibration["cell"], f"{request['task_id']}_{request['attempt_id']}_radonpy_eq21")
    metadata_path = output / "input.snapshot.json"
    _write_json_new(metadata_path, metadata)
    provenance = {
        **build_provenance, "method": METHOD, "source_method": "RADONPY_EQ21", "params": params,
        "stage": "EQ21step_PACKING_COMPRESSION_SAMPLING_COMPLETE_QC_PASS",
        "classical_equilibrium_check": True, "mace_equilibrium_claim": False,
        "equilibration_history": equilibration["history"],
        "total_sampling_steps": equilibration["completed_steps"], "sampling_time_step_fs": 1.0,
        "maximum_sampling_steps": int(params["max_eq_step"] * 1_000_000),
        "original_smiles": request["smiles"], "smiles_sha256": request["smiles_sha256"],
        "atom_type_mass_mapping": mass_mapping, "dependency_versions": api.versions,
        "classical_lammps_sha256": _sha256(Path(params["lammps_exec"])),
        "seed": request["seed"],
        "seed_scope": "Python and NumPy; external engines and RDKit embedding are not guaranteed bitwise reproducible",
    }
    result = {"task_id": request["task_id"], "smiles_sha256": request["smiles_sha256"],
              "input_data": str(final), "mace_elements": elements,
              "snapshot_metadata": str(metadata_path), "preparation_provenance": provenance}
    _write_json_new(output / "prepared.json", result)
    return result


def prepare(request: dict, output_dir: Path, site: dict, *, backend=None) -> dict:
    output, params = _validate(request, output_dir, site)
    output.mkdir(parents=True, exist_ok=True)
    _write_json_new(output / "request_identity.json", request)
    stage = "load_dependencies"
    try:
        api = backend if backend is not None else load_backend(params)
        if api.versions.get("radonpy-pypi") != RADONPY_VERSION:
            raise PreparationError("unsupported RadonPy API version")
        random.seed(request["seed"])
        api.np.random.seed(request["seed"])
        work = output / "radonpy"
        work.mkdir()
        stage = "monomer"
        mol = _require_mol(api.utils.mol_from_smiles(request["smiles"], ez="E", chiral="S"), stage)
        terminal = _require_mol(api.utils.mol_from_smiles("*C", ez="E", chiral="S"), "terminal")
        # The same connection-marker substitution is documented in RadonPy
        # 0.2.11 utils.mol_from_smiles. It is an internal representation only.
        reference_smiles = request["smiles"].replace("[*]", "[3H]").replace("*", "[3H]")
        reference = api.Chem.MolFromSmiles(reference_smiles)
        if reference is None or not mol.HasSubstructMatch(reference, useChirality=True):
            raise PreparationError("UNSUPPORTED_STEREOCHEMISTRY: RadonPy monomer does not preserve the source graph/stereo")
        source_query = api.Chem.MolFromSmarts(request["smiles"])
        if source_query is None:
            raise PreparationError("source graph cannot be checked as a polymer substructure")
        # gen_chi_array('isotactic') is all False in the pinned RadonPy API.
        # It copies, rather than randomly inverting, an explicitly chiral unit.
        effective_tacticity = "isotactic" if "@" in request["smiles"] else params["tacticity"]
        linkers = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "H" and atom.GetIsotope() == 3]
        if len(linkers) != 2:
            raise PreparationError("UNSUPPORTED_CONNECTIONS: repeating unit must have exactly two supported connection points")
        builder_smiles = api.Chem.MolToSmiles(mol, isomericSmiles=True)
        ff = api.GAFF2_mod()
        stage = "monomer_conformation_search"
        mol, _ = api.qm.conformation_search(
            mol, ff=ff, nconf=params["nconf"], dft_nconf=params["dft_nconf"],
            work_dir=str(work), solver_path=params["lammps_exec"],
            psi4_omp=params["psi4_omp"], mpi=params["mpi"], omp=params["omp"],
            gpu=params["gpu"], memory=params["memory_mb"], log_name="monomer")
        mol = _require_mol(mol, stage)
        if not mol.HasSubstructMatch(reference, useChirality=True):
            raise PreparationError("source graph/stereo changed during conformation search")
        stage = "RESP_charges"
        for molecule, name, optimize in ((mol, "monomer", False), (terminal, "terminal", True)):
            if not api.qm.assign_charges(molecule, charge="RESP", opt=optimize,
                                         work_dir=str(work), omp=params["psi4_omp"],
                                         memory=params["memory_mb"], log_name=name):
                raise PreparationError(f"RESP charge assignment failed: {name}")
        stage = "polymerize"
        degree = api.poly.calc_n_from_num_atoms(mol, params["target_atoms_per_chain"], terminal1=terminal)
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
            raise PreparationError("invalid polymerization degree")
        chain = _require_mol(api.poly.polymerize_rw(
            mol, degree, tacticity=effective_tacticity, ff=ff,
            work_dir=str(work), omp=params["omp"], mpi=params["mpi"], gpu=params["gpu"]), stage)
        chain = _require_mol(api.poly.terminate_rw(chain, terminal), "terminate")
        if not chain.HasSubstructMatch(source_query, useChirality=True):
            raise PreparationError("UNSUPPORTED_STEREOCHEMISTRY: terminated chain lacks the original stereochemical repeat substructure")
        stage = "forcefield_assignment"
        if not ff.ff_assign(chain):
            raise PreparationError("GAFF2_mod parameter assignment failed")
        stage = "amorphous_cell"
        cell = _require_mol(api.poly.amorphous_cell(chain, params["chains"], density=params["initial_density"]), stage)
        if cell.GetNumAtoms() != chain.GetNumAtoms() * params["chains"]:
            raise PreparationError("amorphous builder returned an incomplete cell")
        build_provenance = {
            "builder_monomer_smiles": builder_smiles, "terminal_smiles": "*C",
            "polymerization_degree": degree, "atoms_per_chain": chain.GetNumAtoms(),
            "stereochemistry_assumptions": {
                "unspecified_ez": "E", "unspecified_monomer_chiral": "S",
                "requested_tacticity": params["tacticity"], "effective_tacticity": effective_tacticity,
                "explicit_chiral_policy": "isotactic copying without RadonPy chiral inversion",
                "source_monomer_graph_stereo_check": True,
                "source_repeat_substructure_check": True,
                "experimental_microstructure_match": False,
            },
        }
        _write_json_new(output / "build_provenance.json", build_provenance)
        stage = "EQ21step"
        preset = api.eq.EQ21step(cell, work_dir=str(work), solver_path=params["lammps_exec"])
        # v0.2.11 can return a Mol after nonzero LAMMPS exit if output files
        # happen to exist. Independently require all three preset exits to be 0.
        # Its log.lammps is truncated by each invocation, so capture the public
        # exec() return values instead of treating a final log as all-stage proof.
        returncodes = []
        _write_json_new(output / "eq21_execution_started.json", {
            "status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
            "work_dir": str(work), "sampling_steps": int(params["eq_step"] * 1_000_000),
            "sampling_time_step_fs": 1.0,
        })
        try:
            cell = _exec_with_exit_audit(api, lambda: preset.exec(
                temp=params["temperature_k"], press=params["pressure_atm"],
                f_density=params["packing_density"], max_temp=params["max_temperature_k"],
                max_press=params["max_pressure_atm"], time_step=params["time_step_fs"],
                eq_step=params["eq_step"], omp=params["omp"], mpi=params["mpi"],
                gpu=params["gpu"]), returncodes, 3)
        except Exception as exc:
            _write_json_new(output / "eq21_execution_failure.json", {
                "status": "FAILED", "lammps_returncodes": returncodes,
                "error": str(exc), "artifacts": _stage_artifacts(work),
                "ended_utc": datetime.now(timezone.utc).isoformat(),
            })
            raise
        _write_json_new(output / "eq21_execution_complete.json", {
            "status": "COMPLETE", "ended_utc": datetime.now(timezone.utc).isoformat(),
            "lammps_returncodes": returncodes,
        })
        stage = "classical_equilibrium_check"
        equilibrium = equilibrate_bounded(api, cell, params, work, output,
            initial_preset=preset, initial_returncodes=returncodes)
        stage = "final_structure_contract"
        return publish_prepared(request, output, params, api, equilibrium,
            {**build_provenance, "eq21_lammps_returncodes": returncodes})
    except Exception as exc:
        _write_json_new(output / "preparation_failure.json", {
            "task_id": request["task_id"], "smiles_sha256": request["smiles_sha256"],
            "attempt_id": request["attempt_id"], "stage": stage,
            "exception_type": type(exc).__name__, "error": str(exc),
            "prepared": False,
        })
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--site", required=True)
    args = parser.parse_args(argv)
    try:
        request_path = _absolute(args.request, "request")
        site_path = _absolute(args.site, "site")
        request = json.loads(request_path.read_text(encoding="utf-8"))
        site = json.loads(site_path.read_text(encoding="utf-8"))
        prepare(request, _absolute(args.output_dir, "output_dir"), site)
    except Exception as exc:
        print(f"PREPARATION_FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
