#!/bin/bash
#SBATCH -J ghelps_tune
#SBATCH -o results/logs/%x.o%A_%a
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 2:0:0
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --array=0-101

# Usage: sbatch [--array=0-N] scripts/hpc/tune.sbatch [config]   (default configs/tune.yaml)
# One array task per (graph universe, model, graph condition) from configs/tune.yaml (102 tasks; submit_all.sh sets --array from task_list());
# a variant config (configs/tune_time.yaml, 24 tasks) adds the label set and feature set to each task;
# each task tunes both losses. All tasks share results/tuning/optuna.db, so resubmitting
# resumes unfinished studies. If configs/tune.yaml's model list changes, update --array
# (the task list is printed by: uv run python -c "from tune import task_list; print(task_list())").

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/hpc/env.sh   # GHELPS_DATA on scratch

CONFIG="${1:-configs/tune.yaml}"

uv sync --frozen
read -r UNIV MODEL COND LABELS FEATS < <(uv run python -c \
  "from tune import task_list; print(*task_list('${CONFIG}')[${SLURM_ARRAY_TASK_ID}])")
VARIANT=()
if [[ -n "${LABELS:-}" ]]; then VARIANT=(--labels "${LABELS}" --feats "${FEATS}"); fi
echo "tuning ${MODEL} on ${UNIV}, condition ${COND} ${LABELS:-} ${FEATS:-}"
uv run python tune.py "${CONFIG}" --universes "${UNIV}" --models "${MODEL}" --conditions "${COND}" ${VARIANT[@]+"${VARIANT[@]}"} \
  --device cuda
