#!/usr/bin/env bash
# EXAMPLE ONLY: a collaborator MUST provide their own compatible MACE launcher.
#SBATCH --job-name=polymer-density
#SBATCH --array=1-1691%4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=5-00:00:00
set -euo pipefail
# Add the site's account/partition, environment activation, MPI and CUDA setup.
# Do not use the CRC allocation parser under Slurm.
cd /absolute/path/to/mace-density-smiles1691
exec /absolute/path/to/mace-environment/bin/python -B -m polymer_batch.cli run \
  --site /absolute/path/to/mace-density-smiles1691/site.local.json \
  --work-root /absolute/path/to/private_smiles1691_results \
  --task-index "${SLURM_ARRAY_TASK_ID:?array allocation required}" --confirm-run YES
