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

# hydra.run.dir/hydra.sweep.dir, checkpoint_path, and MLflow's tracking dir
# all resolve through the same ${aa_root:} resolver (src/anytimeacquisition/
# utils/paths.py), which defaults to the repo clone's own location -- on
# LUIS that's under $HOME (see CLAUDE.md). LUIS's storage docs are explicit
# that $HOME must not hold data used by compute jobs (slow NFS, tiny quota),
# and that $PROJECT isn't even mounted on compute nodes, only login/transfer
# -- so outputs/, multirun/ (which is also where hydra-submitit-launcher's
# own per-job .submitit/ log dir nests, since it defaults to
# ${hydra.sweep.dir}/.submitit/%j), models/, and mlruns/ all need somewhere
# else. $BIGWORK (InfiniBand, all nodes incl. compute, no backup -- treat as
# scratch, see docs/OPEN_QUESTIONS.md #6) is the only option that's both
# fast and reachable from every job. Redirecting AA_PROJECT_ROOT moves all
# four together in one step, rather than four separate env vars drifting
# out of sync; AA_MLFLOW_DIR is still honored on top of this if you ever
# want mlflow data to live somewhere different from the rest (e.g. $PROJECT
# for archival) -- see configs/callbacks/mlflow.yaml.
# Exported here (not just in train.sbatch) so both dispatch paths below get
# it: sbatch inherits the submitting shell's environment by default, and
# the multirun path calls python directly in this same shell.
: "${BIGWORK:?\$BIGWORK is not set -- scripts/submit.sh targets LUIS, is this its login node?}"
export AA_PROJECT_ROOT="${AA_PROJECT_ROOT:-${BIGWORK}/AnytimeAcquisition}"
mkdir -p "${AA_PROJECT_ROOT}"

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
