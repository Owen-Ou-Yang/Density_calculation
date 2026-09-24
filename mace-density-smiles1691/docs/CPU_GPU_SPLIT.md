# CPU preparation first, GPU density only after preparation passes

The legacy `run` / `stage=all` path performs CPU preparation inside its GPU
allocation. Setting the classical adapter's `gpu=0` does not release scheduler
GPU resources. Use two separate allocations to remove that waste. This is a
resource and evidence-handoff change, not a new scientific protocol.

```text
CPU queue, zero GPUs
  --stage prepare
  -> PREPARED_QC_PASS + verified prepared structure
  -> offline GPU planning selects only verified ready tasks
GPU queue, existing MACE launcher
  --stage density --prepared-from <fixed CPU attempt>
  -> a new density attempt -> existing MACE QC -> final result or honest failure
```

The helper never calls qsub/sbatch, chains jobs, or automatically retries. In
particular, it does not queue a GPU job that waits inside its allocation for
CPU preparation. Render GPU submissions only after CPU results are available.
No real cluster qualification is established by local fake-process tests.

## 1. Configure independent resource profiles

Use `examples/cluster.prepare.sge.json` or `cluster.prepare.slurm.json` for the
CPU phase. Replace the CPU queue/partition placeholder with the site's approved
CPU queue; do not guess or reuse a GPU queue. Keep the private profiles and all
results outside the entire repository. The CPU profile has:

- `stage: "prepare"`, `gpu_count: 0`, and an explicit positive `cpu_slots`.
- A CPU-only `environment_setup`, classical interpreter/LAMMPS paths and CPU
  budgets appropriate for RadonPy/Psi4. It does not require MACE/CUDA/model files
  to exist on the preparation node.
- Its own CPU walltime, concurrency, memory policy, log directory and work root.
  Psi4's memory setting is not a scheduler memory reservation.

Generate a CPU-only site without installing MACE or downloading its checkpoint
on the preparation node:

```bash
python -B tools/configure_site.py --stage prepare \
  --prep-python /absolute/prep-env/bin/python \
  --classical-lammps /absolute/classical-lammps/bin/lmp \
  --prep-mpi 16 --prep-omp 1 --psi4-omp 16 \
  --output /shared/private/cpu-site.json
```

Adjust CPU/memory settings to the approved CPU allocation. The helper records
paths and budgets; it does not launch preparation. CPU setup must not execute
the GPU launcher. Keep all CPU outputs on persistent shared storage accessible
to the later GPU nodes.

For the GPU phase, use `examples/cluster.density.sge.json` or
`cluster.density.slurm.json`, with `stage: "density"` and `prepared_work_root`
pointing to the CPU work root. Keep
`work_root` as a separate GPU-results directory and use a separate log directory
and GPU environment setup. The density site needs `density_argv` and `density`
with the same agreed checkpoint/head/dtype and qualified launcher. The existing
3/4-GPU × four-threads-per-rank requirements are unchanged.

```bash
python -B tools/configure_site.py --stage density \
  --mace-python /absolute/mace-env/bin/python \
  --model /absolute/checkpoints/mace-mh1.model \
  --mace-launcher /absolute/qualified-launcher.sh \
  --output /shared/private/gpu-site.json
```

Add the existing `--runtime-dependency` entries for the qualified launcher when
needed. The default `--stage all` remains available for legacy combined sites.

## 2. Run CPU preparation in an approved CPU allocation

Offline check and render:

```bash
python -B tools/cluster.py check --profile /shared/private/prepare.sge.json
python -B tools/cluster.py render --profile /shared/private/prepare.sge.json \
  --output-dir /shared/private/submissions/prepare-001
```

Review the generated `submission.json` and `submit-command.txt` before submitting
with the site's approved mechanism. The CPU submission must contain no GPU
resource request. A direct single-task equivalent, **inside a CPU allocation**:

```bash
python -B -m polymer_batch.cli run --stage prepare --task-index 1 \
  --site /shared/private/cpu-site.json --work-root /shared/private/prepared \
  --confirm-run YES
```

Classical preparation and its 5 ns continuation/QC are unchanged. Successful
preparation terminates with `PREPARED_QC_PASS`; it does not report a MACE density,
full pipeline completion, or production eligibility. Failed/interrupted tasks
are not silently retried, and repeating completed preparation does not rerun it.

## 3. Render a GPU array for ready tasks only

After the CPU phase, run offline `check` / `render` with the density profile.
Within its selected catalog range, the renderer includes only verified
`PREPARED_QC_PASS` attempts. Other states are reported as excluded, not as
successful densities. A damaged receipt claiming preparation success is an
error, not a reason to skip verification. No ready tasks means no GPU bundle.

```bash
python -B tools/cluster.py check --profile /shared/private/density.sge.json
python -B tools/cluster.py render --profile /shared/private/density.sge.json \
  --output-dir /shared/private/submissions/density-ready-001
```

The GPU array uses compact indices `1..N_ready`, mapped in the frozen plan to
the original catalog IDs and exact prepared-parent attempt paths. A scheduler
array index is therefore not necessarily a catalog index in this mode. Preserve
the mapping in `submission.json`; do not join results by scheduler index alone.
The rendered worker does not discover new CPU results or choose a newer attempt
at execution time. Parent receipt/input hashes are checked again before model
execution. Render a new non-overwriting bundle to include later CPU successes.

Direct single-task equivalent, **inside a GPU allocation**:

```bash
python -B -m polymer_batch.cli plan --stage density --task-index 1 \
  --site /shared/private/gpu-site.json \
  --prepared-from /shared/private/prepared/S000001/attempt_0001
python -B -m polymer_batch.cli run --stage density --task-index 1 \
  --site /shared/private/gpu-site.json --work-root /shared/private/mace-results \
  --prepared-from /shared/private/prepared/S000001/attempt_0001 --confirm-run YES
```

The GPU run copies verified prepared inputs and parent evidence into a **new**
attempt. The original CPU attempt stays unchanged. It does not repeat DFT,
building, packing or classical equilibration. It starts the usual MACE
initialization and transition because no MACE dynamics have yet been performed
on this newly prepared parent. This differs from `--continue-density-from`, which
continues an already executed MACE transition checkpoint; do not combine them.

## 4. Cost and qualification limits

Split allocation removes GPU reservation during CPU preparation. It does not
make MACE faster, reduce the required effective sample count, or guarantee that
a polymer will build/converge. There may be a second queue wait between phases.
Report CPU core-hours separately from GPU-hours; GPU-hours for the split path
start at the density allocation, not the CPU job's start time.

The split GPU examples request **192 hours (8 days)**; the separate CPU
preparation examples remain at 144 hours. These are requested resource budgets,
not a guarantee that a queue accepts them or that every polymer finishes within
them. Confirm the GPU queue permits 192 hours before submission. The legacy
combined examples retain their earlier 144-hour request and are not the split
workflow. Choose a valid site walltime from measured throughput and the full
allowed sampling budget, with startup, analysis and I/O margin;
this patch does not add cross-job recovery of an interrupted arbitrary target
branch or automatically resubmit an expired allocation.

Use the existing final `COMPLETE_QC_PASS` plus `integrity=VERIFIED` requirement.
Preparation success, scheduler exit zero, code publication or green CPU tests
are not a real end-to-end MACE acceptance result. Qualify one full split task on
the actual CPU/GPU queues before expanding a representative batch.
