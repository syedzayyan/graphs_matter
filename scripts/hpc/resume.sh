#!/bin/bash
# Re-run experiment grids without re-tuning (e.g. after a code fix), then summaries +
# analysis + figures. Finished runs are skipped. From the repo root:
#   bash scripts/hpc/resume.sh [experiment ...]   (default: all four)
set -euo pipefail
mkdir -p results/logs
EXPS=("$@"); [ ${#EXPS[@]} -eq 0 ] && EXPS=(main waves membrane nonsense_labels nonsense_feats)
declare -A SHARDS=([main]=32 [waves]=8 [membrane]=8 [nonsense_labels]=4 [nonsense_feats]=4)
deps=""
for e in "${EXPS[@]}"; do
  id=$(sbatch --parsable --array=0-$((${SHARDS[$e]:-8} - 1)) scripts/hpc/run.sbatch configs/exp_${e}.yaml)
  deps="${deps}:${id}"; echo "${e}: ${id}"
done
summ=$(sbatch --parsable --dependency=afterok${deps} scripts/hpc/summarise.sbatch "${EXPS[@]}")
echo "summarise+analysis+figures: ${summ}"
