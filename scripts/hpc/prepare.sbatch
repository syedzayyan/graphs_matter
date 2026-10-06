#!/bin/bash
#SBATCH -J ghelps_prepare
#SBATCH -o %x.o%j
#SBATCH -p compute
#SBATCH -n 8
#SBATCH -t 12:0:0
#SBATCH --mem-per-cpu=4G

# Fetch all raw sources into $GHELPS_DATA (scratch, see env.sh; ~2.5 GB, the IntAct archive is filtered then deleted)
# and build data/processed. Idempotent: anything already present is skipped.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/hpc/env.sh   # GHELPS_DATA on scratch
uv sync --frozen
uv run python main.py build
