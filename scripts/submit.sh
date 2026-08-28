#!/bin/bash
# Submit a training run to SLURM, tied to a clean git commit.
#
# Single run (sbatches scripts/slurm/train.sbatch):
#   scripts/submit.sh anytimeacquisition.pipelines.train_pfn experiment=pfn_smoke_xdim2
#
# Parallel sweep across seeds/other list args (routes through Hydra's
# submitit launcher, which submits one SLURM job per combination):
#   scripts/submit.sh anytimeacquisition.pipelines.train -m seed=0,1,2,3
#
# First argument is always the pipeline module to run (this changed from an
# earlier version that hardcoded anytimeacquisition.pipelines.train --
# train_pfn.py needed the same dispatch, and there'll be more pipelines).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: working tree is dirty. Commit before submitting a tracked run." >&2
  echo "(the pipeline itself enforces this too at run time; this is just a fast" \
       "pre-check so a dirty repo doesn't waste a place in the SLURM queue)" >&2
  exit 1
fi

mkdir -p logs

MODULE="$1"
shift

multirun=false
for arg in "$@"; do
  if [[ "$arg" == "-m" || "$arg" == "--multirun" ]]; then
    multirun=true
  fi
done

if $multirun; then
  # submitit itself calls sbatch per job -- runs here on the login node
  # against the already-synced .venv, not inside a job.
  echo "submitting multirun via hydra-submitit-launcher..."
  source .venv/bin/activate
  python -m "$MODULE" hydra/launcher=submitit_slurm "$@"
else
  echo "submitting single run via sbatch..."
  sbatch scripts/slurm/train.sbatch "$MODULE" "$@"
fi
