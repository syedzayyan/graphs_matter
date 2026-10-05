#!/bin/bash
# prepare (fetch + build, once) -> tune (array, one task per model) -> headline grid.
# Each stage only starts if the previous one succeeded. Run from the repo root:
#   bash scripts/hpc/submit_all.sh [experiment config]
set -euo pipefail
CONFIG="${1:-configs/exp_headline.yaml}"
prep=$(sbatch --parsable scripts/hpc/prepare.sbatch)
tune=$(sbatch --parsable --dependency=afterok:${prep} scripts/hpc/tune.sbatch)
run=$(sbatch --parsable --dependency=afterok:${tune} scripts/hpc/run_headline.sbatch "${CONFIG}")
echo "prepare=${prep}  tune=${tune} (array)  run=${run}"
