#!/usr/bin/env bash
# EXAMPLE ONLY: edit installation paths, resources and local environment first.
#$ -S /usr/bin/bash
#$ -q gpu@@zabaras_rtx6k
#$ -l gpu_card=4
#$ -pe smp 16
#$ -l h_rt=144:00:00
#$ -t 1-1691
#$ -tc 4
#$ -j y
set -euo pipefail

export MACE_GPU_COUNT=4
export MACE_THREADS_PER_RANK=4
export CUDA_HOME=/absolute/path/to/cuda
export IMPI_MPIRUN=/absolute/path/to/intel-mpi/bin/mpirun
export IMPI_LIBRARY_PATH=/absolute/path/to/intel-mpi/lib/release:/absolute/path/to/intel-mpi/lib
export LAMMPS_RTX6K_IMPI=/absolute/path/to/lammps-build
export MACE_CONDA_PREFIX=/absolute/path/to/mace-environment
export THERMAL_PYTHON=/absolute/path/to/mace-environment/bin/python
export MACE_NVIDIA_SMI=/usr/bin/nvidia-smi
# Load your site's RadonPy/classical and MACE runtime libraries here.
# The two Python environments may be separate (configured in site.local.json).
cd /absolute/path/to/mace-density-smiles1691
exec "${THERMAL_PYTHON}" -B -m polymer_batch.cli run \
  --site /absolute/path/to/mace-density-smiles1691/site.local.json \
  --work-root /absolute/path/to/private_smiles1691_results \
  --task-index "${SGE_TASK_ID:?array allocation required}" --confirm-run YES
