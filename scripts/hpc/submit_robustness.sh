#!/bin/bash
# Robustness runs for the report, on the data the main pipeline prepared (run jobs fetch and
# build only what is new: BioGRID 2009 and the dated graphs). Run from the repo root:
#   bash scripts/hpc/submit_robustness.sh
#  1. retune RF and the GP on the time split per label set and feature set
#     (configs/tune_time.yaml), then rerun the 6.4 comparison with those params
#     (configs/exp_novelty_tuned.yaml)
#  2. the time split on a dated graph (BioGRID Dec 2009) and a density control
#     (configs/exp_dated_graph.yaml), in parallel with 1
#  3. summarise both, analysis, DeLong, summary.md
set -euo pipefail
mkdir -p results/logs
N_TUNE=$(uv run python -c "from tune import task_list; print(len(task_list('configs/tune_time.yaml')))" 2>/dev/null | tail -1)
tune=$(sbatch --parsable --array=0-$((N_TUNE - 1)) scripts/hpc/tune.sbatch configs/tune_time.yaml)
tuned=$(sbatch --parsable --array=0-1 --dependency=afterok:${tune} --kill-on-invalid-dep=yes \
  scripts/hpc/run.sbatch configs/exp_novelty_tuned.yaml)
dated=$(sbatch --parsable --array=0-1 scripts/hpc/run.sbatch configs/exp_dated_graph.yaml)
summ=$(sbatch --parsable --dependency=afterok:${tuned}:${dated} --kill-on-invalid-dep=yes \
  scripts/hpc/summarise.sbatch novelty_tuned dated_graph)
echo "tune=${tune} (${N_TUNE} tasks)  novelty_tuned=${tuned}  dated_graph=${dated}  summarise=${summ}"
