# Polymer density compute collaboration: 1,691 original SMILES

Code and an **unfiltered catalog of 1,691 exact SMILES strings**, selected from
the owner's experimental-density catalog. **No experimental density values,
old simulation results, initial structures or model weights are distributed.**
This package computes density with an existing MACE-MH-1 / omol / float32
potential; it does not train a model.

Repository entrypoints: [English](../README.md) / [中文](../README.zh-CN.md).
Commands below run from this package subdirectory. Full real end-to-end
acceptance is still pending; see the dated [validation record](VALIDATION.md).

We welcome compute collaborators. One shared task list makes it possible to
divide the work across people, machines or scheduler arrays without editing
1,691 input files. Configure your site's installed tools once, then run the
whole list or an assigned subset. Return the resulting records privately.

## Identity and coverage

`inputs/smiles.csv` has exactly three columns:

| Column | Meaning |
|---|---|
| `task_id` | Stable catalog index `S000001` through `S001691` |
| `smiles` | Exact original string, including stereo/linker notation |
| `smiles_sha256` | SHA-256 of that exact string's UTF-8 bytes |

Only exact duplicate strings were removed, in first-appearance order. No
canonicalization, trimming, chemical validation or applicability filtering was
performed before export. Different strings remain separate even if they might
represent equivalent chemistry. These S IDs are new catalog IDs, **not** the
old P IDs and must never be joined by row number to an unrelated file. Match
future results using `task_id` plus `smiles_sha256` and the original catalog.

There is no special treatment or packaged geometry for the former 15 polymers.
If their SMILES occur in this catalog they are ordinary entries here.

## What the batch runs on a collaborator's machine

1. **RadonPy preparation**: interpret the repeating unit, construct and cap a
   chain, build an amorphous cell, assign GAFF2_mod/RESP and run the classical
   EQ21step preparation preset. The included adapter targets RadonPy 0.2.11.
   If the first 5 ns sampling segment finishes normally but fails the original
   equilibrium QC, it continues from its saved state in 5 ns Additional
   segments, up to 50 ns total by default. It stops at the first QC PASS;
   execution, nonfinite-value or structure errors stop immediately.
2. **MACE density screening**: use that newly prepared structure with the
   density execution core and unchanged model, timestep and QC thresholds.
   ESS-only rejection can receive bounded additional sampling
   under the [sampling continuation policy](#bounded-mace-sampling-continuation).
3. **Record the outcome**: preserve original SMILES identity, parameters,
   preparation evidence, run manifest, QC and per-task logs.

These preparation operations have **not** already been performed for the
catalog. They are part of the collaborator's computation. The batch attempts
every selected task; unsupported chemistry, preparation errors, OOM and QC
failures are recorded individually and do not stop unrelated tasks. A failed
task is not a density result and does not cause automatic threshold relaxation.
This is batch automation, **not a guarantee of 1,691 successful densities**.

## Quick start: no scientific software needed

From this folder, with Python 3.11+:

```bash
python -B -m polymer_batch.cli plan
python -B tools/check_public_bundle.py
python -m pip install -r requirements-test.txt
python -B -m pytest -q -p no:cacheprovider tests
```

Planning and these tests do not run chemistry, MACE, LAMMPS, GPU kernels or
scheduler clients. Tests use synthetic fixtures and fake child executables.
See [validation scope](VALIDATION.md).

## Configure once

Install the separate [preparation and MACE environments](docs/ENVIRONMENT.md).
Obtain the agreed model from its authorized source. The package neither
downloads it nor supplies your compute-account details.

```bash
python -B tools/configure_site.py \
  --prep-python /absolute/path/to/radonpy-env/bin/python \
  --mace-python /absolute/path/to/mace-env/bin/python \
  --classical-lammps /absolute/path/to/classical-lammps/lmp \
  --model /absolute/path/to/MACE-MH-1_omol_float32.model-mliap_lammps.pt \
  --mace-launcher /absolute/path/to/site-mace-launcher.sh \
  --runtime-dependency /absolute/path/to/mace-lammps/lmp \
  --runtime-dependency /absolute/path/to/mpi/bin/mpirun \
  --output /absolute/path/to/private_config/site.local.json
```

Use the supplied CRC launcher only for its intended Grid Engine / Intel MPI
environment. Other clusters must configure a compatible site launcher. The
helper writes absolute command arrays and the model hash; it does not test the
scientific runtime or start a process. Adjust allocated CPU and memory settings
once in `site.local.json`, and preserve that configuration with returned results.
Keep this configuration outside the entire Git repository.

## Cluster collaborator entrypoint

Prefer the [HPC runbook](docs/CLUSTER_RUNBOOK.md) to editing raw array scripts.
Copy `examples/cluster.slurm.json` or `examples/cluster.sge.json` outside the
repository and configure your site paths, allocation and assigned range.

```bash
python -B tools/cluster.py check --profile /absolute/path/to/private_config/cluster.local.json
python -B tools/cluster.py render --profile /absolute/path/to/private_config/cluster.local.json \
  --output-dir /absolute/path/to/private_submissions/qualification-001
```

The helper never submits. It emits one coordinator per array task and a reviewed
submission argument list, leaving MPI launch to the existing site launcher.
Use the generated full command, not bare submission of `job.sh`. Start with
one task, then scale within the agreed GPU/CPU/storage budget. Offline validation
does not establish runtime or scientific qualification.

For the reference CRC hardware, separate CPU/GPU resource requests and a
measured-versus-extrapolated time budget, see
[CRC configuration and compute cost](docs/CRC_RESOURCES_AND_COST.md).

## Run the entire catalog

For CPU preparation without holding GPU resources, use the
[two-allocation workflow](docs/CPU_GPU_SPLIT.md). It preserves the commands below
as the legacy `--stage all` interface. New `--stage prepare` and `--stage density`
commands separate resource use and require a verified prepared-parent handoff.

Inside a permitted compute allocation, with runtime environments configured:

```bash
python -B -m polymer_batch.cli run \
  --site /absolute/path/to/private_config/site.local.json \
  --work-root /absolute/path/to/private_smiles1691_results \
  --confirm-run YES
```

This processes all 1,691 tasks **sequentially within this allocation**. It does
not start 1,691 GPU jobs at once. Never run real calculations on a login node.
An exit code of 1 can mean some tasks failed or failed QC even though the batch
continued through the list; inspect the per-task statuses.

For throughput use scheduler arrays or non-overlapping shards:

```bash
# One array element, using a 1-based catalog index:
python -B -m polymer_batch.cli run --task-index 42 \
  --site /absolute/path/to/site.local.json \
  --work-root /absolute/path/to/private_smiles1691_results --confirm-run YES

# Worker 0 of 8, with its own assigned compute allocation:
python -B -m polymer_batch.cli run --shard-index 0 --shard-count 8 \
  --site /absolute/path/to/site.local.json \
  --work-root /absolute/path/to/private_smiles1691_results --confirm-run YES
```

Run shard indices 0 through 7 to cover the list exactly once. Alternatively,
`--start 1 --stop 100` selects an inclusive range. Do not mix overlapping
assignments across separate machines; filesystem locks cannot coordinate
independent disks. Shared work roots require a filesystem with working POSIX
file locks. [CRC and Slurm array templates](examples/) are examples to adapt,
not automatic submissions. Limit concurrency according to the resource owner.

## Continue and collect

Repeating the same batch command skips completed, hash-verified tasks and
continues unstarted ones. Failed or interrupted tasks are **not** silently
rerun. To explicitly retry an assigned failed/incomplete task, add
`--task-index N --retry-failed`; a new attempt directory is created and the
old evidence stays intact. This is task-level continuation, not an MD restart
resume. Within a new preparation attempt, bounded classical equilibration
extensions are automatic; they are not scheduler resubmissions. Every segment
has separate immutable execution/QC records. QC-negative completed MACE tasks
remain QC-negative; an eligible transition needs the explicit continuation
selection below to create another attempt.

```bash
python -B -m polymer_batch.cli status \
  --work-root /absolute/path/to/private_smiles1691_results
python -B tools/export_summary.py \
  --work-root /absolute/path/to/private_smiles1691_results \
  --output /absolute/path/to/private_smiles1691_summary.csv
```

Each summary retains all catalog tasks. Unstarted/failed tasks have missing
density, not zero. Completed target-window estimates retain their QC label.
Do not publish the result CSV or generated run folders in this code repository.

## Bounded MACE sampling continuation

The density PILOT protocol prospectively includes `sampling_continuation` with
`increment_ps=25`, `max_transition_ps=100` and `max_density_ps=100`.
Continuation is eligible only when a window completes normally and
`minimum_effective_samples` is its **only** failed QC check. Thresholds remain
10 effective samples for transition and 20 for target sampling.

| Stage | Initial sampling | QC after each 25 ps extension | Cumulative sampling cap |
| --- | --- | --- | --- |
| 305 K transition | 50 ps | Fresh 25 ps window, assessed independently | 100 ps |
| 300 K density | 25 ps after the existing 0.1 ps ramp and 5 ps equilibration | All post-equilibration samples: 25, 50, 75, then 100 ps | 100 ps |

Each extension starts at the verified preceding endpoint and runs only 25 ps
of new dynamics. A transition extension receives QC on its fresh 25 ps window;
it does not pool the earlier failed transition samples or initial relaxation.
At 300 K, the sampling dataset instead grows cumulatively after the prescribed
equilibration, and QC is recomputed at 25, 50, 75 and 100 ps. The target ramp
and equilibration are not repeated for each extension. The budget counts the
initial and additional sampling, including that consumed by a parent attempt.
This is a finite sampling budget, not a promise that any assessment will pass.

Runtime failures, OOM, nonfinite values, invalid structures, temperature
violations, and other or mixed QC failures stop immediately. The sampling
outcomes `NEEDS_MORE_SAMPLING`, `SAMPLING_BUDGET_EXHAUSTED` and
`SAMPLING_QC_FAILED` do not indicate QC PASS. All failed windows and their
execution/QC records, restart endpoints, identities and budget history remain
available for inspection; a later passing assessment does not rewrite them.
The policy is defined for new continuation runs; it does not retroactively
reclassify an earlier failed result. Each cumulative target assessment also
keeps its own QC record and the identities of the contributing sample segments.

For an existing eligible terminal transition, validate one explicit parent
attempt offline, using the matching site configuration:

```bash
python -m polymer_batch.cli plan --task-index 1 \
  --site /absolute/path/site.json \
  --continue-density-from /absolute/path/old_results/S000001/attempt_0001
```

Then run within a permitted compute allocation:

```bash
python -m polymer_batch.cli run --task-index 1 \
  --site /absolute/path/site.json \
  --work-root /absolute/path/private_results \
  --continue-density-from /absolute/path/old_results/S000001/attempt_0001 \
  --confirm-run YES
```

This creates a new attempt, preserving the parent. It verifies parent/request,
task/SMILES, snapshot and model identities, copies verified prepared inputs,
and skips preparation and MACE initialization before continuing from the saved
transition endpoint. The selector currently accepts transition endpoints only;
it is not an entrypoint for an already terminal 300 K target branch. Offline
planning does not execute chemistry, MD or a scheduler, and does not establish
that the installed GPU runtime or scientific calculation will pass.

Within a running attempt, eligible extensions use the current allocation.
Neither this selector nor budget exhaustion triggers `qsub`, `sbatch` or
automatic resubmission. Confirm the old worker has ended and allocate enough
remaining walltime before running an explicit continuation. The cluster array
renderer does not select a parent attempt; use this single-task command inside
the allocation. Do not combine it with a whole-catalog retry.

The full trajectory remains subject to the existing safety gates. A future
burn-in analysis would need its own stated protocol; this revision does not
trim early samples after seeing QC results or change QC thresholds or safety
policy. Cumulative target analysis follows the sampling rule defined above.

## Scientific scope

This release delegates preparation to the collaborator; it does not certify
chemical identities or model coverage. Builder assumptions (chain length,
capping, tacticity, packing, temperature history) are recorded but are not
automatically matched to experimental samples. Original strings are preserved
even when the builder uses a separate internal representation.

The MACE stage remains a single-packing **PILOT / screening** calculation:
fixed-box minimization, 1 ps NVT at 305 K, 50 ps transition NPT at 305 K, then
0.1 ps ramp, 5 ps equilibration and 25 ps sampling at 300 K / 1.01325 bar;
timestep 0.25 fs. Only the ESS-only continuation described above can add
transition validation windows or cumulative target sampling within its fixed
caps. A blocked transition does not yield a target density. Only the 300 K
sampling branch is summarized. The classical preparation density
is never substituted as a MACE prediction. QC PASS is not experimental accuracy
or full material-level convergence.

No Tg or modulus command is exposed. Some dormant historical types/policy files
remain internal dependencies of the reused execution core.

## Publication and contributions

Upload this directory only, not the enclosing private research project. See
[data boundary](docs/DATA_BOUNDARY.md), [collaboration guide](docs/COLLABORATION.md)
and [license notice](LICENSE_NOTICE.md). This GitHub repository uses the root
[MIT license](../LICENSE); third-party dependencies retain their own terms.
For a standalone copy include LICENSE as well. Keep generated output outside
the whole Git repository. For failures see [troubleshooting](docs/TROUBLESHOOTING.md).

中文：1691 条原始 SMILES 全部保留，不附实验密度、旧结构和已有结果。
协作者配置一次环境后可整批、按编号或按分片运行。建链、装箱、经典平衡和
MACE 计算都在协作者机器上执行；失败逐项记录，不能把失败项当成有效密度。
