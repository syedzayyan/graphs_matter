# Shared environment for all ghelps jobs. Sourced by the sbatch scripts.
# Data (raw downloads ~2.5 GB + built benchmark) lives on scratch, not in the repo.
export GHELPS_DATA=/data/scratch/bty644/ghelps/data
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${GHELPS_DATA}"
