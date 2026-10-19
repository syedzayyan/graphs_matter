#!/bin/bash
#SBATCH -J ghelps_summarise
#SBATCH -o results/logs/%x.o%j
#SBATCH -p compute
#SBATCH -n 8
#SBATCH -t 8:0:0
#SBATCH --mem-per-cpu=4G

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
# Per-graph descriptive tables (sizes, hub bias, membrane enrichment, mediation, study before
# launch, top-k audit, phase ladder) on the same data.
uv run python scripts/analysis.py
# All figure sets per graph and experiment, plus the druggability-vs-membrane comparison. Runs here (not on a laptop) because figure 5 needs the
# gene-level degrees and splits in $GHELPS_DATA that these runs were computed on.
uv run python scripts/figures.py --exps "$@" || echo "figures.py failed; continuing with DeLong" >&2
# Paired DeLong tests (real vs each control, graph model vs no-graph RF), BH-corrected. Slowest
# step, so last: everything above is already written if it runs out of time.
uv run python scripts/delong.py "$@"
# summary.md: every table filled into scripts/report_template.md
uv run python scripts/report.py
