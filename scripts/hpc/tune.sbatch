#!/bin/bash
#SBATCH -J ghelps_tune
#SBATCH -o results/logs/%x.o%A_%a
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH -n 12
#SBATCH --cpus-per-gpu=12
#SBATCH -t 24:0:0
#SBATCH --mem-per-cpu=7500M
#SBATCH --gres=gpu:1
#SBATCH --array=0-101

# One array task per (graph universe, model, graph condition) from configs/tune.yaml (102 tasks; submit_all.sh sets --array from task_list());
# each task tunes both losses. All tasks share results/tuning/optuna.db, so resubmitting
# resumes unfinished studies. If configs/tune.yaml's model list changes, update --array
# (the task list is printed by: uv run python -c "from tune import task_list; print(task_list())").

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/hpc/env.sh   # GHELPS_DATA on scratch

uv sync --frozen
read -r UNIV MODEL COND < <(uv run python -c "from tune import task_list; print(*task_list()[${SLURM_ARRAY_TASK_ID}])")
echo "tuning ${MODEL} on ${UNIV}, condition ${COND}"
uv run python tune.py configs/tune.yaml --universes "${UNIV}" --models "${MODEL}" --conditions "${COND}" --device cuda
