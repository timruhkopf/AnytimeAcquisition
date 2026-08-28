#!/bin/bash
# Single-run SLURM job. Gets submitted via
#   sbatch scripts/slurm/train.sbatch <pipeline-module> <hydra overrides...>
# by scripts/submit.sh — not meant to be sbatch'd directly.
#
# TODO (see docs/OPEN_QUESTIONS.md): fill in partition/account/qos/time for
# your cluster below, or pass them via `sbatch --partition=...` on the
# command line from scripts/submit.sh instead of hardcoding here.
#SBATCH --job-name=anytimeacquisition
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:00:00      # TODO: set a time limit
#SBATCH --partition=         # TODO: set your partition
#SBATCH --gres=gpu:1         # assumes a CUDA run (train_pfn); remove/comment
                              # out for a CPU-only pipeline

set -euo pipefail

# Uses the .venv that was `uv sync --extra ...`'d ahead of time on the login
# node -- NOT `uv run` here. uv run would re-resolve/re-sync on every job,
# which is slow and depends on network access compute nodes often don't
# have; sync once where there's internet, then just use the built venv.
source .venv/bin/activate

MODULE="$1"
shift

python -m "$MODULE" "$@"
