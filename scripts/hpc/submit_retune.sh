#!/bin/bash
# Retune RF and the GP on the time split for each label set and leak-free feature set
# (configs/tune_time.yaml), rerun the 6.4 comparison with those params
# (configs/exp_novelty_tuned.yaml), then summarise, analyse and rebuild summary.md.
# Needs the data the main pipeline prepared (no prepare step). Run from the repo root:
#   bash scripts/hpc/submit_retune.sh
set -euo pipefail
mkdir -p results/logs
N_TUNE=$(uv run python -c "from tune import task_list; print(len(task_list('configs/tune_time.yaml')))" 2>/dev/null | tail -1)
tune=$(sbatch --parsable --array=0-$((N_TUNE - 1)) scripts/hpc/tune.sbatch configs/tune_time.yaml)
run=$(sbatch --parsable --array=0-1 --dependency=afterok:${tune} --kill-on-invalid-dep=yes \
  scripts/hpc/run.sbatch configs/exp_novelty_tuned.yaml)
summ=$(sbatch --parsable --dependency=afterok:${run} --kill-on-invalid-dep=yes scripts/hpc/summarise.sbatch novelty_tuned)
echo "tune=${tune} (${N_TUNE} tasks)  run=${run}  summarise=${summ}"
