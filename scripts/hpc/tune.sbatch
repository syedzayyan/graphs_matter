#!/bin/bash
#SBATCH -J ghelps_tune
#SBATCH -o %x.o%A_%a
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH -n 12
#SBATCH --cpus-per-gpu=12
#SBATCH -t 24:0:0
#SBATCH --mem-per-cpu=7500M
#SBATCH --gres=gpu:1
#SBATCH --array=0-32

# One array task per (model, graph condition) pair from configs/tune.yaml (33 pairs);
# each task tunes both losses. All tasks share results/tuning/optuna.db, so resubmitting
# resumes unfinished studies. If configs/tune.yaml's model list changes, update --array
# (the task list is printed by: uv run python -c "from tune import task_list; print(task_list())").

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/hpc/env.sh   # GHELPS_DATA on scratch

uv sync --frozen
read -r MODEL COND < <(uv run python -c "from tune import task_list; print(*task_list()[${SLURM_ARRAY_TASK_ID}])")
echo "tuning ${MODEL} on condition ${COND}"
uv run python tune.py configs/tune.yaml --models "${MODEL}" --conditions "${COND}" --device cuda
