#!/bin/bash
#SBATCH -J ghelps_headline
#SBATCH -o %x.o%j
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH -n 12
#SBATCH --cpus-per-gpu=12
#SBATCH -t 24:0:0
#SBATCH --mem-per-cpu=7500M
#SBATCH --gres=gpu:1

# Headline grid on one GPU. Finished runs are skipped, so resubmitting resumes.
# Missing data is fetched and built automatically (see ghelps/ensure.py).
# Usage: sbatch scripts/hpc/run_headline.sbatch [config]   (default configs/exp_headline.yaml)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONFIG="${1:-configs/exp_headline.yaml}"

uv sync --frozen
uv run python main.py device
uv run python main.py --device cuda run "${CONFIG}" --workers 4
uv run python main.py summarise "$(basename "${CONFIG}" .yaml | sed 's/^exp_//')"
