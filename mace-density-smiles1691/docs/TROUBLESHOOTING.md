# Troubleshooting without changing scientific conclusions

## Offline checks fail

Run commands from `mace-density-smiles1691/`, not the repository root. Use Python
3.11+ and install `requirements-test.txt` into a CPU test environment. Do not
try to fix CPU test failures by loading a model or submitting a scheduler job.

`check_public_bundle.py` checks exact package files and catalog hashes. A fresh
checkout must include hidden `.gitignore`; web uploads can omit it. If a local
site file/result causes an unexpected-file error, move your configuration/work
outside the public checkout or check a fresh clone. Do not rewrite hashes to
approve unknown data. Intentional source edits need a reviewed new manifest.

The CRC launcher must be executable; Git should record mode `100755`. If a
transport stripped modes, restore them on a non-running source copy. That is
different from changing its bytes. `configure_site.py` rejects non-executable
launchers rather than deferring the error to an expensive compute job.

## Preparation does not converge at 5 ns

Normal engine completion with ordinary QC rejection is not a runtime crash.
The adapter continues from saved state in bounded 5 ns Additional segments,
including the initial 5 ns in the default 50 ns total budget. Check segment
records before interpreting elapsed time. It does not repeat QM, chain building
or packing on each extension. At the budget limit, unresolved QC remains failure;
never substitute the classical density for the MACE result or loosen thresholds.

Nonzero engine exit, nonfinite quantities or damaged structures stop preparation
immediately. Fix only a reproduced execution problem before authorizing a new
attempt. A completed-but-QC-negative MACE task is not automatically retried.

## `status` reports INCOMPLETE

```bash
python -B -m polymer_batch.cli status --task-index 1 \
  --work-root /absolute/path/to/private_results
```

This can mean the attempt is still running, was interrupted, or failed integrity
verification. Inspect the status error/active-claim fields, scheduler state,
stage logs and final receipt together. Absence from `qstat`/`squeue` alone proves
neither success nor failure. Do not launch a duplicate while a worker is active.

Final `COMPLETE_QC_PASS` and `integrity=VERIFIED` qualify a completed PILOT record.
`COMPLETE_QC_FAIL` means the execution completed but the recorded QC did not pass;
it is not an accepted density. Native run manifests and raw-file hashes still
belong in the final evidence review. A process exit code of zero alone is not
the scientific acceptance criterion.

## OOM, allocation or launcher errors

Record CPU-vs-GPU memory error, atom count, allocated GPU type/count, MPI ranks
and the failed stage privately. More GPUs do not guarantee lower peak memory:
domain decomposition, ghost atoms and per-rank model allocations also matter.
Do not silently switch checkpoint/dtype, guess physical device IDs or run outside
the assigned allocation. Validate a site-compatible launcher on one task. The
included CRC Intel MPI launcher is not a generic Slurm launcher.

## The calculation is slow or storage is growing

Preparation is CPU/QM/classical work; the MACE phase has many short time steps.
Their walltimes cannot be inferred from the same physical duration alone. Measure
actual progress per phase on your installation. The supplied one-job examples may
reserve idle GPUs during CPU preparation; this is documented, not hidden parallelism.
Check quota and output growth before a large batch. Do not delete live files or
evidence covered by a receipt. Contact the maintainer before changing storage
cadence, restart semantics or scientific sampling to reduce cost.

## Report a reproducible issue

Include the code commit, relevant software versions, stage, task ID and a short
sanitized error excerpt. Do not publish raw results, trajectories, full site files,
private paths or credentials. Use synthetic fixtures for code regressions.

