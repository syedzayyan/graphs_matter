#!/bin/bash
# Prepare (fetch + build whatever is missing), re-run experiment grids (e.g. after a code fix),
# then summaries + analysis + figures + DeLong.
# Finished runs and finished tuning studies are skipped. From the repo root:
#   bash scripts/hpc/resume.sh [experiment ...]               (default: all six)
#   RETUNE=31,43,46 bash scripts/hpc/resume.sh                re-run those tune array tasks first
set -euo pipefail
mkdir -p results/logs
EXPS=("$@"); [ ${#EXPS[@]} -eq 0 ] && EXPS=(main waves membrane nonsense_labels nonsense_feats novelty)
declare -A SHARDS=([main]=12 [waves]=3 [membrane]=2 [nonsense_labels]=1 [nonsense_feats]=1 [novelty]=4)
# fetch new raw sources and build missing processed data once, before any run job does it
prep=$(sbatch --parsable scripts/hpc/prepare.sbatch)
after="--dependency=afterok:${prep} --kill-on-invalid-dep=yes"; echo "prepare: ${prep}"
if [ -n "${RETUNE:-}" ]; then
  tune=$(sbatch --parsable --array="${RETUNE}" ${after} scripts/hpc/tune.sbatch)
  after="--dependency=afterok:${tune} --kill-on-invalid-dep=yes"; echo "tune (${RETUNE}): ${tune}"
fi
deps=""
for e in "${EXPS[@]}"; do
  id=$(sbatch --parsable --array=0-$((${SHARDS[$e]:-4} - 1)) ${after} scripts/hpc/run.sbatch configs/exp_${e}.yaml)
  deps="${deps}:${id}"; echo "${e}: ${id}"
done
summ=$(sbatch --parsable --dependency=afterok${deps} --kill-on-invalid-dep=yes scripts/hpc/summarise.sbatch "${EXPS[@]}")
echo "summarise+analysis+figures: ${summ}"
