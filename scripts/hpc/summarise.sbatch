#!/bin/bash
#SBATCH -J ghelps_summarise
#SBATCH -o results/logs/%x.o%j
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -t 4:0:0
#SBATCH --mem-per-cpu=8G

# Collect every shard's runs into results/<exp>/results.csv + summary CSVs, then draw
# results/figures/<graph>/ for every graph in the given experiments.
# Usage: sbatch scripts/hpc/summarise.sbatch <experiment name> [<experiment name> ...]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/hpc/env.sh
for exp in "$@"; do
  uv run python main.py summarise "${exp}"
  uv run python main.py summarise "${exp}" --metric auprc
done
# Paired DeLong tests (real vs each control, graph model vs no-graph RF), BH-corrected.
uv run python scripts/delong.py "$@"
# Per-graph descriptive tables (sizes, hub bias, membrane enrichment) on the same data.
uv run python scripts/analysis.py
# All figure sets per graph and experiment, plus the druggability-vs-membrane comparison. Runs here (not on a laptop) because figure 5 needs the
# gene-level degrees and splits in $GHELPS_DATA that these runs were computed on.
uv run python scripts/figures.py --exps "$@"
