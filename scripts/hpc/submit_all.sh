#!/bin/bash
# prepare (fetch + build, once) -> tune (one task per model x graph condition)
#   -> headline + HuRI grids (sharded array jobs) -> summaries.
# Each stage only starts if the previous one succeeded. Run from the repo root:
#   bash scripts/hpc/submit_all.sh
set -euo pipefail
mkdir -p results/logs   # slurm -o paths point here and slurm won't create the directory
N_TUNE=$(uv run python -c "from tune import task_list; print(len(task_list()))" 2>/dev/null | tail -1)
SHARDS=8

prep=$(sbatch --parsable scripts/hpc/prepare.sbatch)
tune=$(sbatch --parsable --array=0-$((N_TUNE - 1)) --dependency=afterok:${prep} scripts/hpc/tune.sbatch)
head=$(sbatch --parsable --array=0-$((SHARDS - 1)) --dependency=afterok:${tune} scripts/hpc/run.sbatch configs/exp_headline.yaml)
huri=$(sbatch --parsable --array=0-$((SHARDS - 1)) --dependency=afterok:${tune} scripts/hpc/run.sbatch configs/exp_huri.yaml)
summ=$(sbatch --parsable --dependency=afterok:${head}:${huri} scripts/hpc/summarise.sbatch headline huri)
echo "prepare=${prep}  tune=${tune} (${N_TUNE} tasks)  headline=${head}  huri=${huri} (${SHARDS} shards each)  summarise=${summ}"
