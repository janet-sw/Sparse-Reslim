#!/usr/bin/env bash
#SBATCH -A csc662
#SBATCH -J sparse-reslim-table1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 12:00:00
#SBATCH -o sparse-reslim-table1-%j.out
#SBATCH -e sparse-reslim-table1-%j.out

set -euo pipefail

module load PrgEnv-gnu
module load "${ROCM_MODULE:-rocm/6.3.1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/table1_sparse_era5_1.40625.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

: "${DATA_ROOT:?Set DATA_ROOT to the ERA5 1.40625-degree dataset directory}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
export TRAINING_SEED="${TRAINING_SEED:-42}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-7}"
export PYTHONNOUSERSITE=1

MASTER_HOST="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_ADDR="${MASTER_ADDR:-${MASTER_HOST}}"
export MASTER_PORT="${MASTER_PORT:-29500}"

cd "${REPO_ROOT}"
srun --kill-on-bad-exit=1 \
  "${PYTHON_BIN}" examples/era5_forecasting_sparsity.py "${CONFIG_PATH}"
