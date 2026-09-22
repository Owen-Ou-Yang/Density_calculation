# Collaborator environment setup

For profile-based Slurm/SGE arrays, resource checks and private paths, follow
the [HPC runbook](CLUSTER_RUNBOOK.md). It does not replace a site-qualified MACE
launcher or turn the CPU requirements file into a GPU environment installer.

## Planning, batch control and tests

Use Python 3.11+ on a POSIX system. Catalog listing, planning and task dispatch
use the standard library. The copied MACE analysis/runtime needs NumPy and
pandas; `requirements-test.txt` also supplies pytest for CPU/fake-process tests.
These files are **not** a complete GPU or preparation environment lockfile.
Nothing in a planning command downloads a package or model.

## Preparation environment

The bundled adapter calls **RadonPy 0.2.11** (`radonpy-pypi==0.2.11`). Install
its supported stack including RDKit, NumPy/SciPy/pandas, MDTraj, Psi4, RESP,
dftd3 and a classical LAMMPS build with molecular force fields, KSPACE, SHAKE
and XTC support. Follow the [RadonPy installation instructions](https://github.com/RadonPy/RadonPy/tree/v0.2.11).
Do not assume that installing a small Python requirements file installs these
compiled/QM/MD dependencies. The collaborator is responsible for their site's
compatible build, license terms and MPI library setup.

Preparation uses the official `mol_from_smiles`, conformation/RESP, polymer
construction and `EQ21step` APIs. The exact API references are linked in
`adapters/radonpy_prepare.py`. This is a new public-input preparation path,
not a claim of reproducing the private historical ADEPT preparation protocol.
The output contract records `RADONPY_EQ21`, not `ADEPT_EQ1_EQ2`.

Default preparation settings are about 600 atoms/chain, six chains, initial
packing density 0.05 g/cm3, methyl terminal groups, 300 K and 1 atm. The
EQ21step preset includes its packing/compression history and five million
sampling steps by default. A normally completed but QC-negative segment is
followed by 5 million-step `Additional` segments until the original QC passes
or the default cumulative limit of 50 million sampling steps is reached.
RadonPy 0.2.11 fixes these NPT sampling steps to 1 fs (5 ns per default chunk,
50 ns total); `time_step_fs` applies to the earlier packing/compression preset.
All settings, segment exit codes, QC metrics and artifact hashes are recorded.
Execution errors, invalid structures or nonfinite results are not retried.
This stage can be substantial work. It does not use experimental density as
a target or loosen QC after seeing results. The loop is never unlimited.

The original SMILES stays unchanged in the task catalog and receipts. The
builder's own representation and requested/effective tacticity are separate
provenance fields. Unspecified stereochemistry and molecular weight cannot be
reconstructed from an experimental density measurement. Do not equate a default
constructed cell with an experimentally characterized material microstructure.

`site.preparation` controls the fixed settings for a compute batch. It accepts
`lammps_exec`, `target_atoms_per_chain`, `chains`, `initial_density`,
`temperature_k`, `pressure_atm`, `packing_density`, `max_temperature_k`,
`max_pressure_atm`, `time_step_fs`, `eq_step`, `max_eq_step`, `tacticity`, `omp`, `mpi`, `gpu`,
`psi4_omp`, `memory_mb`, `nconf`, and `dft_nconf`. See the adapter's `DEFAULTS`
for exact values. `eq_step` and `max_eq_step` are million-step budgets; the
maximum must be a finite exact multiple of the chunk and at least one chunk.
Each is at least 5 million steps, preserving the analysis window. A site may
explicitly choose another finite maximum; the default is 50 million steps.
The supplied helper keeps classical preparation on CPU
(`gpu=0`); it does not silently consume the MACE GPU allocation with a second
parallel model. Choose CPU/memory counts within the scheduler allocation.

## MACE environment

Provide the agreed pretrained **MACE-MH-1 / omol / float32** ML-IAP checkpoint.
The configuration helper hashes it; do not substitute a different head, dtype
or model for missing element coverage. Record an unsupported task instead.
No model weights are supplied or automatically downloaded.

Real MD requires a compatible MACE/PyTorch/cuEquivariance/CUDA and
LAMMPS/MPI/Kokkos/Python/ML-IAP stack. The [official MACE ML-IAP guide](https://mace-docs.readthedocs.io/en/latest/guide/lammps_mliap.html)
describes conversion/build requirements. Model conversion is separate GPU work
and can be architecture-sensitive. Agree converted-model identity with the
coordinator; CPU tests do not qualify a new GPU installation.

Keep preparation and MACE in separate Python environments if their dependencies
conflict. `prepare_argv` and `density_argv` each contain an absolute executable;
both receive the same exact task identity. Site-specific library paths and MPI
activation still need to be arranged inside the compute job. Do not depend on
a login-node working directory or ship private environment archives.

## Reference launcher and other schedulers

`launchers/crc_intelmpi_3or4gpu.sh` retains the existing 3/4-GPU Intel MPI
behavior. On its intended CRC environment set:

```bash
export MACE_GPU_COUNT=4
export MACE_THREADS_PER_RANK=4
export CUDA_HOME=/absolute/path/to/cuda
export IMPI_MPIRUN=/absolute/path/to/intel-mpi/bin/mpirun
export IMPI_LIBRARY_PATH=/absolute/path/to/intel-mpi/lib/release:/absolute/path/to/intel-mpi/lib
export LAMMPS_RTX6K_IMPI=/absolute/path/to/lammps-build
export MACE_CONDA_PREFIX=/absolute/path/to/mace-environment
export THERMAL_PYTHON=/absolute/path/to/mace-environment/bin/python
export MACE_NVIDIA_SMI=/usr/bin/nvidia-smi
```

The launcher maps the scheduler's allocated GPU identifiers to GPU UUIDs; it
does not invent allocation 0/1/2/3. It is not a generic Slurm launcher. Slurm
or another MPI implementation requires a collaborator-provided launcher that
preserves the intended model/runtime and handles that site's allocation.

Array templates only select a row and call the same batch entry point. They do
not implement auto-submission/resubmission or promise a queue/walltime valid at
another site. Each example array element performs both preparation and density,
so a GPU allocation may be idle during CPU preparation. A more efficient
separate CPU/GPU scheduling policy is a site adaptation, not an undocumented
optimization in this release.

Use absolute paths without whitespace for package, model, inputs and output on
compute nodes. Validate the real installation before committing a large resource
budget. The bounded-extension patch has local fake-backend tests. Its real
end-to-end acceptance remains pending; do not label it validated yet.

## Before a large allocation

Start with the [offline checks](../README.md#quick-start-no-scientific-software-needed),
then one real task. Confirm the actual child interpreters, imported libraries,
allocated GPU count and checkpoint identity using the run records. A successful
import is not an end-to-end test. Preserve the software versions with private
results; this repository does not ship a portable binary environment.

Plan for storage as well as GPU memory: classical trajectories and frequent
MACE restart files can grow substantially. Choose a dedicated work root outside
the entire Git repository on storage appropriate for the batch. Do not remove
live attempt files or receipt-bound artifacts to free space. There is no automatic
garbage collection. A different output cadence needs a separately reviewed code
and protocol change; do not edit a running attempt.

See [troubleshooting](TROUBLESHOOTING.md) for the distinction between ordinary
equilibration QC rejection, runtime failure, and still-running attempts.
