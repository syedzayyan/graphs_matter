#!/bin/bash
# prepare (fetch + build, once) -> tune (one task per graph x model x condition)
#   -> all experiment grids (sharded array jobs) -> summaries + analysis + figures.
# Each stage only starts if the previous one succeeded. Run from the repo root:
#   bash scripts/hpc/submit_all.sh
set -euo pipefail
mkdir -p results/logs   # slurm -o paths point here and slurm won't create the directory
EXPS=(main membrane nonsense_labels nonsense_feats)
declare -A SHARDS=([main]=32 [membrane]=8 [nonsense_labels]=4 [nonsense_feats]=4)
N_TUNE=$(uv run python -c "from tune import task_list; print(len(task_list()))" 2>/dev/null | tail -1)

prep=$(sbatch --parsable scripts/hpc/prepare.sbatch)
tune=$(sbatch --parsable --array=0-$((N_TUNE - 1)) --dependency=afterok:${prep} scripts/hpc/tune.sbatch)
deps=""
for e in "${EXPS[@]}"; do
  id=$(sbatch --parsable --array=0-$((SHARDS[$e] - 1)) --dependency=afterok:${tune} scripts/hpc/run.sbatch configs/exp_${e}.yaml)
  deps="${deps}:${id}"; echo "${e}: ${id} (${SHARDS[$e]} shards)"
done
summ=$(sbatch --parsable --dependency=afterok${deps} scripts/hpc/summarise.sbatch "${EXPS[@]}")
echo "prepare=${prep}  tune=${tune} (${N_TUNE} tasks)  summarise+analysis+figures=${summ}"
