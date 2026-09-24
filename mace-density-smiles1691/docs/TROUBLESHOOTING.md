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

## MACE completes but effective sampling is insufficient

Read the failed checks, not only the engine's exit code. Only a sole
`minimum_effective_samples` failure is eligible for bounded density PILOT
continuation. A transition must still meet 10 effective samples; the 300 K
target window must meet 20. No threshold is reduced.

The initial 305 K transition is 50 ps and may receive 25 ps validation windows
up to 100 ps cumulative transition sampling. Each fresh transition extension
is assessed independently, without pooling failed parent transition samples.
The target branch keeps its 0.1 ps ramp, 5 ps equilibration and first 25 ps
sampling window. It adds only 25 ps of new dynamics per extension, then uses
all cumulative post-equilibration samples for QC at 25, 50, 75 and 100 ps.
Every extension starts from the verified preceding endpoint. These distinct
rules are prescribed prospectively; they do not rewrite an earlier result.
Old attempts, sample segments, QC assessments, restarts and consumed-budget
history remain immutable, including earlier ESS-negative target assessments.

`NEEDS_MORE_SAMPLING` denotes eligible insufficient sampling within the budget;
`SAMPLING_BUDGET_EXHAUSTED` means no permitted window remains;
`SAMPLING_QC_FAILED` means the QC result is not eligible for ESS-only
continuation. None is QC PASS. Runtime/OOM, nonfinite, structure or temperature
errors and other/mixed QC failures stop immediately. They must not be treated
as insufficient sampling merely because ESS also failed.

For a completed eligible transition, use the explicit single-task
[`--continue-density-from` plan/run commands](../README.md#bounded-mace-sampling-continuation).
Planning validates the selected parent and site offline; execution requires a
permitted allocation and an ended parent worker. A new attempt copies verified
prepared inputs and skips preparation/initialization. The current selector
accepts transition endpoints only, not an already terminal 300 K target branch.
Do not use `--retry-failed` as a substitute for selecting that endpoint or
manually edit receipts, model identities or consumed budgets.

The policy does not resubmit a scheduler job. Do not launch a new task merely
because its predecessor vanished from the scheduler. Inspect terminal evidence,
active claims and the remaining sampling/resource budgets first.

## Can early transient samples be trimmed to pass QC?

Not under this revision. The existing full-trajectory safety gates continue to
check runtime, finite values, structure and temperature. A late stable-looking
region does not erase an earlier safety violation. Fresh transition windows
and cumulative post-equilibration target assessments are defined in advance;
neither permits post-hoc cuts of a failed window. Any future burn-in analysis
needs a separate stated protocol and cannot relabel this run's failed QC as PASS.

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

## CPU preparation holds GPU resources

`preparation.gpu=0` selects CPU classical execution but cannot release GPUs
reserved by the enclosing scheduler job. Use the [CPU/GPU split](CPU_GPU_SPLIT.md):
`stage=prepare`, zero GPUs, an approved CPU queue and its own CPU-only setup;
then render `stage=density` only for verified `PREPARED_QC_PASS` parents.
Do not edit or reclassify old running allocations. CPU success is not final
density success; failed/missing preparation must not start a GPU calculation.
