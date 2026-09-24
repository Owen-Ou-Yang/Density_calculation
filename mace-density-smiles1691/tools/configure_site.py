"""Create a local site configuration once; no models or subprocesses run."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise argparse.ArgumentTypeError("supply an existing absolute file path")
    return path.resolve()


def python_path(value):
    # A venv's bin/python is often a symlink. Executing its resolved target can
    # silently leave the selected environment and lose RadonPy/MACE packages.
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise argparse.ArgumentTypeError("supply an existing absolute Python path")
    return path.absolute()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "prepare", "density"), default="all",
                        help="generate only the dependencies used by this allocation")
    parser.add_argument("--prep-python", type=python_path)
    parser.add_argument("--mace-python", type=python_path)
    parser.add_argument("--classical-lammps", type=file_path)
    parser.add_argument("--model", type=file_path)
    parser.add_argument("--mace-launcher", type=file_path)
    parser.add_argument("--runtime-dependency", type=file_path, action="append", default=[])
    parser.add_argument("--prep-mpi", type=int, default=1)
    parser.add_argument("--prep-omp", type=int, default=1)
    parser.add_argument("--psi4-omp", type=int, default=1)
    parser.add_argument("--psi4-memory-mb", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    prepare = args.stage in ("all", "prepare")
    density = args.stage in ("all", "density")
    required = (["prep_python", "classical_lammps"] if prepare else []) + (
        ["mace_python", "model", "mace_launcher"] if density else [])
    for name in required:
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required for stage {args.stage}")
    for name in (name for name in required if name != "model"):
        if not os.access(getattr(args, name), os.X_OK):
            parser.error(f"{name} must be executable")
    if any(getattr(args, name) < 1 for name in ("prep_mpi", "prep_omp", "psi4_omp", "psi4_memory_mb")):
        parser.error("CPU and memory settings must be positive")
    def command(python, adapter):
        return [str(python), "-B", str(ROOT / "adapters" / adapter), "--request", "{request}",
                "--output-dir", "{output_dir}", "--site", "{site}"]
    config = {}
    if prepare:
        config["prepare_argv"] = command(args.prep_python, "radonpy_prepare.py")
        config["preparation"] = {"lammps_exec": str(args.classical_lammps),
                        "target_atoms_per_chain": 600, "chains": 6, "initial_density": 0.05,
                        "temperature_k": 300.0, "pressure_atm": 1.0,
                        "tacticity": "atactic", "mpi": args.prep_mpi, "omp": args.prep_omp,
                        "gpu": 0, "psi4_omp": args.psi4_omp, "memory_mb": args.psi4_memory_mb}
    if density:
        config["density_argv"] = command(args.mace_python, "mace_density.py")
        config["density"] = {"model_path": str(args.model), "model_sha256": sha(args.model),
                    "launcher_path": str(args.mace_launcher),
                    "runtime_dependencies": [str(p) for p in args.runtime_dependency]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(config, out, indent=2, sort_keys=True)
        out.write("\n")
    print(json.dumps({"site_config": str(args.output.resolve()), "stage": args.stage,
                      "model_sha256": config.get("density", {}).get("model_sha256"),
                      "processes_started": 0, "scientific_preparation_performed": False}, indent=2))


if __name__ == "__main__":
    main()
