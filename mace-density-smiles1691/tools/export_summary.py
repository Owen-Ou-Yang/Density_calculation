"""Export every task's verified status and optional computed density privately."""
import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from polymer_batch.catalog import load_catalog
from polymer_batch.cli import _summary, _work_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    work = _work_root(args.work_root, create=False)
    output = args.output.expanduser().resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("summary contains private results; write it outside this package")
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["task_id", "smiles_sha256", "status", "attempt_id", "execution_status", "qc_status",
               "density_g_cm3", "density_standard_error_g_cm3", "error"]
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for task in load_catalog(ROOT / "inputs/smiles.csv"):
            writer.writerow(_summary(task, work / task["task_id"]))
    print(f"Wrote {output}; missing/failed tasks are not replaced by zero density.")


if __name__ == "__main__":
    main()
