# Polymer density from SMILES with MACE-MH-1

[中文说明](README.zh-CN.md) · [HPC runbook](mace-density-smiles1691/docs/CLUSTER_RUNBOOK.md) · [Run guide](mace-density-smiles1691/README.md) · [Environment](mace-density-smiles1691/docs/ENVIRONMENT.md) · [Validation](mace-density-smiles1691/VALIDATION.md)

A compute-collaboration pipeline for **1,691 exact polymer SMILES strings**.
It prepares an atomistic cell with RadonPy/GAFF2_mod, then runs a
**MACE-MH-1 / omol / float32** density PILOT. It uses a pretrained model;
it does not train one.

```text
Original SMILES → chain / packing / classical equilibration
               → unchanged classical QC must pass
               → MACE initialization and NPT transition
               → 300 K density sampling → verified receipt and QC
```

**Release status: engineering tests are available; full real SMILES-to-MACE
acceptance is not yet complete.** A single-input continuation passed
classical QC and reached MACE, but transition sampling did not satisfy the
effective-sample requirement. This is not a final validated density.
This catalog is not chemically prescreened, and not every entry is guaranteed
to build or converge. See the dated [validation record](mace-density-smiles1691/VALIDATION.md).

## Try the interface without a GPU

On a POSIX system with Python 3.11+, start at the repository root:

```bash
git clone https://github.com/Owen-Ou-Yang/Density_calculation.git
cd Density_calculation
python3 -m venv .venv
source .venv/bin/activate
cd mace-density-smiles1691
python -m pip install -r requirements-test.txt
python -B tools/check_public_bundle.py
python -B -m polymer_batch.cli plan --task-index 1
python -B -m pytest -q -p no:cacheprovider tests
```

These checks use synthetic data and fake scientific executables. They do not
run MACE, LAMMPS, chemistry, GPU kernels, or a scheduler. The GitHub Actions
workflow performs CPU checks only; a green CI badge is not MD validation.

## Run real calculations

For cluster collaborators, start with the **[standard HPC runbook](mace-density-smiles1691/docs/CLUSTER_RUNBOOK.md)**.
It separates scientific settings (`site.local.json`) from scheduler/resources
(`cluster.local.json`) and site module setup (`environment.sh`), all private.
From the package directory:

```bash
python -B tools/cluster.py check --profile /shared/project/density-private/cluster.local.json
python -B tools/cluster.py render --profile /shared/project/density-private/cluster.local.json \
  --output-dir /shared/project/density-private/submissions/qualification-001
```

The helper supports single-node **Slurm and SGE arrays**, checks CPU budgets,
paths and checkpoint identity, and produces `job.sh`, `submission.json` and the
exact `submit-command.txt`. It never submits. Use that command after site review,
not bare `sbatch job.sh` / `qsub job.sh`; resource options live in the command.
The starter profiles select one task with concurrency one. Increase the assigned
range only after a real task qualifies your installation. A site-specific MACE
launcher is still required; scheduler portability is not GPU-stack portability.

1. Install separate compatible [preparation and MACE environments](mace-density-smiles1691/docs/ENVIRONMENT.md).
   GPU software, model weights and compiled scientific dependencies are not
   included, and `requirements-test.txt` does not install them.
2. Use [`tools/configure_site.py`](mace-density-smiles1691/tools/configure_site.py)
   to bind absolute interpreter, checkpoint and launcher paths. Keep the site
   file private. Do not replace the agreed checkpoint, head or dtype.
3. Validate **one task inside a permitted compute allocation** before a large
   batch. Follow the [complete run commands](mace-density-smiles1691/README.md#configure-once).
4. Assign non-overlapping task ranges/shards to collaborators. The default
   whole-catalog command runs sequentially within one allocation, not 1,691
   simultaneous jobs. Prefer the profile-based renderer; the older raw
   [CRC/Slurm examples](mace-density-smiles1691/examples/) remain illustrative
   and need site-specific configuration.
5. Preserve successes and failures. Share results privately using the
   [handoff checklist](mace-density-smiles1691/docs/COLLABORATION.md#private-handoff).

Real preparation and MACE sampling can take days for a single system depending
on its size, convergence and hardware. Budget walltime, CPU/GPU memory and disk
space from a representative task; there is no universal hours-per-polymer promise.

### CRC resources and estimated CPU/GPU time

The legacy combined [SGE profile](mace-density-smiles1691/examples/cluster.sge.json)
requests `gpu@@zabaras_rtx6k`, 4 GPUs, `smp` 16 CPU slots and a 144-hour
walltime limit. The MACE launcher uses 4 MPI ranks with 4 threads per rank.
The default profile selects one task with concurrency one; it is not an
instruction to launch the whole catalog. Confirm GPU model/memory and scheduler
policy on the actual allocated node. The 144 hours is a limit, not an estimate.

The recommended [split GPU profile](mace-density-smiles1691/examples/cluster.density.sge.json)
instead requests **192 hours (8 days)**, with CPU preparation in a separate
allocation. Confirm that the chosen queue permits this request; it is not a
catalog-wide runtime guarantee.

For a historical **approximately 3,600-atom** reference cell on **4 x Quadro
RTX 6000** GPUs, use this conditional planning example:

| Phase | Resources in the recommended split | Elapsed time | Allocated resource-hours |
| --- | --- | --- | --- |
| Classical preparation | 16 CPU slots, zero GPUs | About **43 h**, observed on the historical CPU/GPU node | About **690 CPU-slot-hours** |
| MACE density | 4 GPUs and 16 CPU slots; 4 MPI ranks x 4 threads | About **59–150 h**, extrapolated from measured transition speed | About **236–600 GPU-hours**, plus **940–2,400 CPU-slot-hours** |

The MACE range spans initial-window success through the full bounded sampling
budget. It is **not a measured successful end-to-end runtime**: only about 37 h
of initialization/transition was actually completed, and the final 300 K density
stage was not reached. CPU-queue performance may differ. Queue waits, extra I/O
and analysis overhead are additional; QC success is not guaranteed. See the
[CRC configuration and compute-budget guide](mace-density-smiles1691/docs/CRC_RESOURCES_AND_COST.md)
for the reference software, arithmetic, measured/estimated distinction and limits.

There is no calibrated automatic GPU-hour estimator for all 1,691 inputs.
Report **allocated GPU-hours = allocated GPU count × running wall-clock hours**
separately from **MACE-stage GPU-hours = GPU count × MACE-stage hours**. The
legacy single-allocation pipeline reserves GPUs during CPU classical
preparation as well; allocated hours do not measure GPU utilization or necessarily
equal the site's billing charge. Queue waiting is not execution time.
From a representative task, estimate extra sampling as
`GPU count × measured hours/ps × additional ps`, keeping preparation, initialization,
equilibration, failed attempts and storage budgets explicit. Different cell
sizes, chemistries and convergence histories require separate measurements.

For large batches, use the new [CPU preparation / GPU density split](mace-density-smiles1691/docs/CPU_GPU_SPLIT.md).
The legacy `stage=all` path still reserves GPUs during preparation. In the split
path, the CPU array requests **zero GPUs**, and a GPU array is rendered only for
finished, hash-verified `PREPARED_QC_PASS` tasks. Neither stage submits the next
one automatically. CPU preparation success is not a completed MACE density.

**Scale readiness:** array/sharding and bounded continuation are implemented,
but the new continuation has not yet passed a real complete SMILES-to-300 K
density acceptance run. Use single-task qualification, then a small representative
batch before a large assigned range. CPU tests and the catalog size are not
evidence that all entries will complete or yield production-quality densities.

## What is public?

| Included | Not included |
| --- | --- |
| Code, docs, scheduler examples, synthetic tests | Model weights and installed environments |
| 1,691 original SMILES with stable IDs and hashes | Experimental values and private source tables |
| Fixed PILOT protocol and QC policies | Real calculated densities, structures, trajectories and logs |

Only exact-string duplicates were removed. No chemical canonicalization or
experimental-label filtering was applied. Match returned results by task ID
**and** original SMILES hash, never by an old polymer ID or unrelated row order.
Keep output directories **outside the entire repository**, not only outside
the package subdirectory. See [data boundary](mace-density-smiles1691/docs/DATA_BOUNDARY.md).

## Interpretation

Density PILOT sampling has a prospective bounded continuation policy: only an
isolated `minimum_effective_samples` rejection can trigger 25 ps of additional
dynamics. After the initial 50 ps at 305 K, transition extensions are assessed
as fresh 25 ps windows, with a 100 ps total transition cap. At 300 K,
post-equilibration samples accumulate for QC at 25, 50, 75 and 100 ps; each
extension runs only the next 25 ps. Thresholds remain 10 transition / 20 target
effective samples. Runtime, nonfinite, structure, temperature or other/mixed
QC failures stop continuation. Previous QC records are preserved unchanged.

An existing eligible failed transition can be selected explicitly for a new
attempt; see [continuation commands](mace-density-smiles1691/README.md#bounded-mace-sampling-continuation).
The parent and its failed windows remain immutable. No scheduler submission or
resubmission is automatic. This revision changes engineering behavior; it does
not establish a real accepted density or change full-trajectory safety gates.

The final reported quantity is the **MACE 300 K sampling density**, never the
classical preparation density. `COMPLETE_QC_PASS` plus `integrity=VERIFIED`
qualifies a completed PILOT record; it does not establish experimental accuracy,
production eligibility, or validation of all 1,691 materials. Missing or failed
tasks remain missing/failed; do not replace them with zeros or loosen QC.

No Tg or modulus calculation is exposed by this batch interface.

## Contribute

Compute offers, portability reports and tested code improvements are welcome.
Start with [CONTRIBUTING.md](CONTRIBUTING.md), use a sanitized issue, and arrange
assignments before consuming a large allocation. Never post credentials, full
site configs or private scientific outputs in issues/PRs.

The maintainer has selected the repository's [MIT license](LICENSE). Third-party
software and model weights retain their own terms; see [license notice](mace-density-smiles1691/LICENSE_NOTICE.md).
