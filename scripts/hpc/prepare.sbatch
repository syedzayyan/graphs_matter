#!/bin/bash
#SBATCH -J ghelps_prepare
#SBATCH -o %x.o%j
#SBATCH -p compute
#SBATCH -n 8
#SBATCH -t 12:0:0
#SBATCH --mem-per-cpu=4G

# Fetch all raw sources (~2.5 GB incl. the IntAct human archive, filtered then deleted)
# and build data/processed. Idempotent: anything already present is skipped.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
uv sync --frozen
uv run python main.py build
