# HPC collaborator runbook

Use this guide to configure a Slurm or Grid Engine installation, qualify one
catalog task, and then run an agreed subset of the 1,691 inputs. Commands run
from `mace-density-smiles1691/` with Python 3.11+. Real calculations belong
inside an approved compute allocation. Never run preparation or MD on a login
node.

The cluster helper checks local configuration and renders a submission bundle.
It does **not** call `sbatch`, `qsub`, SSH, a scientific executable, or a retry
service. A successful check or render is not scheduler acceptance, GPU runtime
validation, or a completed density calculation. Full real SMILES-to-MACE
acceptance remains pending in the [validation record](../VALIDATION.md).

## 1. Agree the installation and assignment

Have the local cluster administrator or resource owner review the queue or
partition, account, walltime, GPU request, CPU layout, memory policy, MPI
launcher and environment setup. The generated script is a starting point for
that site's supported resource conventions; it cannot discover them.

Install the [preparation and MACE environments](ENVIRONMENT.md), and obtain the
agreed checkpoint through its authorized source. Model weights, credentials,
installed environments and private site files are not supplied. Keep the
MACE-MH-1 / omol / float32 checkpoint and scientific protocol unchanged. A
missing element or runtime incompatibility does not authorize switching model,
head, dtype or QC thresholds.

Assign non-overlapping catalog indices with the coordinator. Index `1` selects
`S000001`, and index `1691` selects `S001691`; ranges are inclusive. Preserve
`task_id`, exact original `smiles`, and `smiles_sha256`. These IDs are not the
private historical P IDs. Do not join results using unrelated row order.

## 2. Put private files and outputs outside the checkout

Choose absolute paths visible from every relevant compute node. For example:

```text
/shared/project/Density_calculation/             public code checkout
/shared/project/density-private/
    site.local.json                            scientific commands and model hash
    cluster.local.json                         scheduler and resource profile
    environment.sh                             reviewed site environment setup
    logs/                                      existing scheduler log directory
    work/                                      persistent per-task evidence
    submissions/qualification-001/              new rendered submission bundle
```

All private configuration, environment files, submission bundles, logs and
results must be outside the **entire Git repository**, including its root.
Keep installed environments and model files outside the repository too. Use
paths without whitespace. The logs directory must exist before rendering and
submission: the scheduler opens its output before the job body can create a
directory.

Shared scratch is suitable only when it has enough quota, is retained for the
campaign, and supports POSIX file locks. This workflow has no node-local staging
or copy-back step; do not point `work_root` at ephemeral node-local scratch.
The code, interpreters, model, launcher and setup files must also remain
accessible throughout the job. Locks coordinate workers sharing a filesystem;
they cannot prevent overlap on separate collaborators' disks.

## 3. Configure the scientific commands once

Run the existing helper with your installed paths and a new private output:

```bash
python -B tools/configure_site.py \
  --prep-python /absolute/path/to/radonpy-env/bin/python \
  --mace-python /absolute/path/to/mace-env/bin/python \
  --classical-lammps /absolute/path/to/classical-lammps/lmp \
  --model /absolute/path/to/MACE-MH-1_omol_float32.model-mliap_lammps.pt \
  --mace-launcher /absolute/path/to/site-mace-launcher.sh \
  --runtime-dependency /absolute/path/to/mace-lammps/lmp \
  --runtime-dependency /absolute/path/to/mpi/bin/mpirun \
  --prep-mpi 1 --prep-omp 4 --psi4-omp 4 \
  --psi4-memory-mb 4000 \
  --output /shared/project/density-private/site.local.json
```

The CPU and memory values above are illustrative, not a measured requirement.
Set them within the reviewed allocation. Classical preparation stays on CPU
(`preparation.gpu=0`). Its `mpi * omp` count and the Psi4 `psi4_omp` count must
each fit the allocated CPU budget. Psi4's memory setting is not a total process
or node memory limit; allow room for other scientific libraries and runtime
overhead.

The helper records absolute command arrays and the checkpoint SHA-256. Separate
preparation and MACE interpreters are supported. The site launcher's job is to
start the MACE LAMMPS ranks, preserve the agreed runtime, and use only scheduler
allocated GPUs. Never guess physical GPU IDs.

The supplied [`crc_intelmpi_3or4gpu.sh`](../launchers/crc_intelmpi_3or4gpu.sh)
is specific to its intended CRC Grid Engine / Intel MPI environment. Use it
only there with the documented [environment variables](ENVIRONMENT.md#reference-launcher-and-other-schedulers).
Slurm requires a site-compatible launcher; the CRC allocation parser is not
accepted as a Slurm launcher. The private `environment.sh` is sourced by the
job to load the compatible modules, library paths and launcher variables for
both stages. Review it locally and keep secrets out of it. Do not rely on
interactive shell startup files or a login node's activated environment.

## 4. Create and check a cluster profile

Copy the matching template to a private location and replace its placeholders:

```bash
cp examples/cluster.slurm.json /shared/project/density-private/cluster.local.json
# For Grid Engine, copy examples/cluster.sge.json instead.
cp examples/environment.sh.example /shared/project/density-private/environment.sh
```

The setup example deliberately stops until you replace it with a site-reviewed
environment. Create `work/`, `logs/` and the `submissions/` parent directory
before checking/rendering; the new submission leaf must not exist.

For qualification, set `task_start=1`, `task_stop=1`, and `max_concurrent=1`, or
use another single index explicitly assigned by the coordinator.

| Profile field | Meaning |
| --- | --- |
| `scheduler` | `slurm` or `sge` |
| `name`, `queue` | Job name and site partition/queue |
| `account` | Site account/project; an empty string omits it |
| `walltime` | Positive `HH:MM:SS` budget, within the site's limit |
| `gpu_count` | 3 or 4 GPUs per array element on one node |
| `threads_per_rank` | 4; CPU budget is `gpu_count * 4` per element |
| `memory_gb` | Slurm total node memory; `null` for Grid Engine |
| `max_concurrent` | Maximum simultaneous elements in this array |
| `task_start`, `task_stop` | Inclusive catalog indices in `1..1691` |
| `driver_python` | Absolute path to the Python 3.11+ batch driver interpreter |
| `site_config` | Absolute path to the generated private site JSON |
| `environment_setup` | Absolute path to the reviewed private shell setup file |
| `work_root`, `log_dir` | Absolute private shared-storage paths |
| `sge_pe` | Grid Engine only; site parallel environment, default `smp` |
| `sge_gpu_resource` | Grid Engine only; site GPU resource name, default `gpu_card` |
| `sge_memory_resource` | Grid Engine only; explicit site-approved memory resource string, default `null` |

Slurm `memory_gb` is a host-memory request, not GPU VRAM or per-rank memory.
Grid Engine memory resources and whether they apply per slot or per job are
site-defined: `memory_gb=null` does not reserve or validate a portable memory
budget. If required by your site, set `sge_memory_resource` to its reviewed
resource expression, for example `mem_free=8G` only if the administrator confirms
that name, amount and per-slot/per-job interpretation. Absence produces a
warning and no memory request. The Grid Engine resource names and parallel
environment must match your installation, and that parallel environment must
keep this job on one node. Resolve any additional site-required options before
submission.

```bash
python -B tools/cluster.py check \
  --profile /shared/project/density-private/cluster.local.json
```

`check` is read-only. It checks the profile, paths and executable references,
site configuration, checkpoint hash, CPU-only classical preparation and thread
budgets. It does not import the scientific stack, execute the interpreters,
source the setup script, inspect a live allocation, or validate queue access.
It cannot establish that an executable called Python is the intended Python
version or that the model/runtime fits GPU memory. Resolve all reported errors
and perform real qualification below.

## 5. Render, review, and submit one qualification task

Use a new output directory for each submission bundle:

```bash
python -B tools/cluster.py render \
  --profile /shared/project/density-private/cluster.local.json \
  --output-dir /shared/project/density-private/submissions/qualification-001
```

The bundle contains `job.sh`, `submission.json`, and `submit-command.txt`.
Review the script, manifest and exact submission command against the assignment
and approved resources; preserve them with the private run evidence. Scheduler
resource arguments are in the emitted command and its manifest argument array,
not directives in `job.sh`. **Use the emitted command; bare `sbatch job.sh` or
`qsub job.sh` omits the reviewed resource request.** Rendering does not submit.

Keep referenced configurations and setup files fixed from review through
execution. The script checks the array index and single-node CPU allocation,
then captured hashes for the site configuration, environment setup, checkpoint,
catalog, protocol and launcher before sourcing setup. Slurm requires the batch
step's `SLURM_GPUS_ON_NODE` to match the request. SGE requires a unique token
count from `SGE_HGR_TASK_<resource>` / `SGE_HGR_<resource>` (matching channels
when both are present). An SGE site without this resource-map interface needs
a reviewed site adaptation; the wrapper does not guess GPU IDs. The bundled
CRC launcher also gets a required-variable/executable check after setup, before
preparation. Other launchers remain site-qualified. These checks do not validate the
contents of trusted site code or replace real runtime qualification. Grid
Engine's command begins with `-clear` and explicit options, without `-V` or
`-cwd`. Slurm uses `--export=ALL` plus the explicit setup file; review the
submission shell's environment too.

Each array element starts **one plain-Python batch driver** for its catalog
index. The inner site launcher owns MPI parallelism. Do not wrap the batch
driver in `mpirun`, `mpiexec`, or a multi-task `srun`: doing so can start several
copies of preparation and the whole task. The allocation is one node with
3 or 4 GPUs and respectively 12 or 16 CPU threads. CPU preparation and GPU
MACE execute in that same allocation, so reserved GPUs may be idle during
preparation. This release does not split those stages into separate jobs.

Slurm log names use `%A_%a` to distinguish the array and element. Grid Engine
receives the existing log directory and supplies its scheduler-specific array
filenames; it does not interpret Slurm's `%A`/`%a`. Consult the official
[Slurm array guide](https://slurm.schedmd.com/job_array.html),
[Slurm sbatch manual](https://slurm.schedmd.com/sbatch.html), and
[Grid Engine qsub manual](https://gridscheduler.sourceforge.net/htmlman/htmlman1/qsub.html)
alongside your local scheduler documentation.

Once the one-task environment and resource request have been reviewed locally,
the collaborator manually executes the command in `submit-command.txt` **once**
and records the returned job/array ID. Do not pipe the file to a shell as part
of an automatic loop. An ambiguous submission response needs scheduler/accounting
inspection before any further submission. No helper command submits or retries
a job.

Let this task exercise the actual preparation, launcher and MACE density path.
Record software versions, GPU type/count, CPU use, peak memory, disk growth and
elapsed time per stage. A successful import, scheduler acceptance, first GPU
step, classical QC PASS or an in-progress MACE run is not full qualification.
For a successfully completed density task, require `COMPLETE_QC_PASS` with
`integrity=VERIFIED`, and inspect the preparation and native MACE manifests,
QC and logs. A rejected chemistry or failed task remains a valid reported
outcome, but does not qualify the unexercised end-to-end path.

## 6. Expand to the agreed range within its budget

After one real task qualifies the installation, review its cost with the
resource owner before expanding. Set only the assigned `task_start`/`task_stop`
and concurrency, check again, and render into a new submission directory.
Execute its emitted submit command once after review. Larger or chemically
different inputs may still need more time or memory than the qualification
task.

For `N = task_stop - task_start + 1`, concurrency `C`, and GPUs per element `G`:

- At most `min(N, C)` elements from this array run simultaneously.
- Their simultaneous GPU request is `min(N, C) * G` and CPU request is
  `min(N, C) * G * 4` threads.
- An allocation ceiling for one pass is `N * G * walltime_hours` GPU-hours,
  including GPU reservation during CPU preparation. Retries or additional
  arrays add to that budget; scheduler billing rules may differ.

For example, 100 tasks with 4 GPUs each and concurrency 2 can reserve 8 GPUs
and 32 CPU threads at once. At a 120-hour walltime, the requested ceiling is
48,000 GPU-hours. This arithmetic is a resource budget, not a runtime estimate.
Check array-size and maximum-index limits, per-user limits, filesystem quota,
and the aggregate concurrency across **all** collaborators and arrays. Slurm's
maximum array index must accommodate the selected catalog indices even if
only a small high-index range is selected. Walltime does not promise model
performance or that every polymer will finish.

The helper renders contiguous index ranges. For manually assigned shards,
the [batch interface](../README.md#run-the-entire-catalog) supports
`--shard-index K --shard-count M` inside separately approved allocations.
Indices are zero-based for shards and one-based for catalog tasks. Do not
combine overlapping shards and arrays or start a whole-catalog driver inside
every array element. Each shard processes its selected tasks sequentially;
its walltime must cover that work.

## 7. Interpret progress, interruption, and scientific QC

```bash
python -B -m polymer_batch.cli status \
  --work-root /shared/project/density-private/work
```

Read the task status together with scheduler state, stage logs, active claims
and receipts. `INCOMPLETE` can mean a live attempt, interruption, or an integrity
problem. It is not enough to declare scientific failure or authorize a duplicate
job. Absence from `squeue`/`qstat`, a scheduler exit code, or elapsed walltime
alone proves neither successful density nor failed science.

Ordinary classical QC rejection after a normally completed initial 5 ns segment
can trigger saved-state `Additional` segments in 5 ns chunks, bounded at 50 ns
total by default. This does not repeat packing/QM and is not a scheduler retry.
The original QC must pass; runtime errors, nonfinite values and invalid
structures stop immediately. At the finite budget limit, unresolved QC remains
failure. Do not relax thresholds or shorten the protocol to fit a queue limit.

The batch skips verified completed tasks when rerun and does not silently retry
failed or incomplete tasks. Any explicit `--retry-failed` requires a reviewed
task-specific reason, confirmation that no worker is active, and a new resource
budget. It creates a new attempt while retaining old evidence; it is not a
general MD checkpoint-resume facility. Preserve completed QC-negative outcomes.
See [troubleshooting](TROUBLESHOOTING.md) before deciding whether another attempt
is appropriate.

`COMPLETE_QC_PASS` plus verified integrity qualifies a completed **PILOT /
screening** record. It does not establish experimental agreement, production
convergence or validity across the catalog. The reported quantity is the MACE
300 K sampling density; classical preparation density is never a replacement.
Experimental comparison belongs to the coordinator using private labels.

## 8. Return evidence privately

Follow the [private handoff checklist](COLLABORATION.md#private-handoff). Include
the code commit and package/catalog identities; exact task IDs and SMILES hashes;
submission bundle, profile and site configuration; reviewed environment setup
and software versions; scheduler job IDs and resource records; summary output;
and complete attempt receipts, requests, prepared structures/provenance, native
run manifests, QC and logs. Include archive SHA-256 values and retain the model
hash without redistributing model weights or credentials.

Keep successful, QC-negative, failed, interrupted and unstarted tasks
distinguishable. Preserve manifest-bound paths and byte identity; moving an
archive is not permission to rewrite its provenance. Arrange a private transfer
with the coordinator. Do not commit generated results, structures, trajectories,
logs, profiles or site details to this repository, and do not attach them to
public issues or pull requests.
