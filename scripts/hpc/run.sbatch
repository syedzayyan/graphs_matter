#!/bin/bash
#SBATCH -J ghelps_run
#SBATCH -o results/logs/%x.o%A_%a
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH -n 12
#SBATCH --cpus-per-gpu=12
#SBATCH -t 24:0:0
#SBATCH --mem-per-cpu=7500M
#SBATCH --gres=gpu:1
#SBATCH --array=0-7

# One experiment grid split into disjoint shards, one per array task (the array size is the
# shard count). Finished runs are skipped, so resubmitting resumes.
# Usage: sbatch [--array=0-N] scripts/hpc/run.sbatch configs/exp_headline.yaml

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/hpc/env.sh   # GHELPS_DATA on scratch
CONFIG="${1:?usage: sbatch scripts/hpc/run.sbatch <experiment config>}"
N_SHARDS=$(( SLURM_ARRAY_TASK_MAX + 1 ))

uv sync --frozen
uv run python main.py --device cuda run "${CONFIG}" --workers 4 --shard "${SLURM_ARRAY_TASK_ID}/${N_SHARDS}"
