#!/bin/bash
#SBATCH -J ghelps_summarise
#SBATCH -o results/logs/%x.o%j
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=8G

# Collect every shard's runs into results/<exp>/results.csv + summary CSVs.
# Usage: sbatch scripts/hpc/summarise.sbatch <experiment name> [<experiment name> ...]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/hpc/env.sh
for exp in "$@"; do
  uv run python main.py summarise "${exp}"
  uv run python main.py summarise "${exp}" --metric auprc
done
