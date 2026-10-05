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
#SBATCH --array=0-14

# One array task per model in configs/tune.yaml (15 models, both losses each).
# All tasks share results/tuning/optuna.db, so resubmitting resumes unfinished studies.
# Afterwards:  sbatch scripts/hpc/run_headline.sbatch   (main.py picks up best_params.yaml)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODELS=($(uv run python -c "import yaml; print(' '.join(yaml.safe_load(open('configs/tune.yaml'))['models']))"))
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "tuning ${MODEL}"

uv sync --frozen
uv run python tune.py configs/tune.yaml --models "${MODEL}" --device cuda
