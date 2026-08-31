#!/bin/bash
# View LUIS's MLflow file-store dashboard locally, WITHOUT running any
# service on LUIS's shared login node.
#
# LUIS is the actual SLURM cluster (SSH alias `luis`,
# login.cluster.uni-hannover.de) -- shared with many other users. Starting
# a long-lived `mlflow ui` process there the way
# `scripts/mlflow_tunnel_ulysses.sh` does for the personal, single-tenant
# `ulysses` box risks looking like an unexpected service to admins and ties
# up a login-node slot for as long as you're looking at the dashboard.
# Instead, this script:
#   1. rsyncs LUIS's mlruns/ dir down to a local cache
#      (~/.cache/anytimeacquisition/luis_mlruns by default), over the
#      dedicated `luis-transfer` host (transfer.cluster.uni-hannover.de)
#      instead of the login node -- that's what it's there for, and it
#      keeps this off `luis` entirely. Assumes `luis`/`luis-transfer` share
#      the same home-directory filesystem (typical for HPC clusters, not
#      independently verified here -- check first if a sync comes back
#      with nothing new).
#   2. runs `mlflow ui` purely LOCALLY against that synced copy -- no SSH
#      tunnel needed for this part, since nothing remote is running.
#
# Usage:
#   scripts/mlflow_sync_luis.sh                       # one-shot sync + view
#   scripts/mlflow_sync_luis.sh 5001                   # different local port
#   AA_WATCH_INTERVAL=60 scripts/mlflow_sync_luis.sh   # re-sync every 60s
#     in the background while mlflow ui runs, so the dashboard stays close
#     to live; Ctrl-C stops both the re-sync loop and mlflow ui together.
#   AA_REMOTE_MLFLOW_DIR=/some/shared/mlruns scripts/mlflow_sync_luis.sh
# Then open http://localhost:5000 (or your chosen port) locally, once the
# "Listening at: ..." line appears below.
#
# Defaults assume:
#   - `luis`/`luis-transfer` SSH config aliases already set up locally.
#   - the repo cloned at ~/AnytimeAcquisition on LUIS. Different clone
#     path? Override AA_REMOTE_REPO_DIR.
#   - MLflow's tracking dir is the default `${aa_root:}/mlruns` inside that
#     repo clone, i.e. AA_MLFLOW_DIR was NOT set for the SLURM runs you
#     want to look at (docs/OPEN_QUESTIONS.md #6 -- what shared path
#     AA_MLFLOW_DIR should point at on LUIS is still open). If a run DID
#     set it, point AA_REMOTE_MLFLOW_DIR at that same path instead, or this
#     script syncs an empty/wrong directory.
set -euo pipefail

TRANSFER_HOST="${AA_LUIS_TRANSFER_HOST:-luis-transfer}"
REMOTE_REPO_DIR="${AA_REMOTE_REPO_DIR:-\$HOME/AnytimeAcquisition}"
REMOTE_MLFLOW_DIR="${AA_REMOTE_MLFLOW_DIR:-${REMOTE_REPO_DIR}/mlruns}"
LOCAL_CACHE_DIR="${AA_LOCAL_MLFLOW_CACHE:-$HOME/.cache/anytimeacquisition/luis_mlruns}"
LOCAL_PORT="${1:-5000}"
WATCH_INTERVAL="${AA_WATCH_INTERVAL:-0}"

mkdir -p "${LOCAL_CACHE_DIR}"

sync_once() {
  # REMOTE_MLFLOW_DIR may contain a literal, unexpanded `$HOME` (see the
  # default above) -- rsync's ssh transport runs the remote side through
  # the remote user's shell, so it expands there, same as the ulysses
  # tunnel script's remote command.
  rsync -az --delete --stats -e ssh "${TRANSFER_HOST}:${REMOTE_MLFLOW_DIR}/" "${LOCAL_CACHE_DIR}/"
}

echo "syncing ${TRANSFER_HOST}:${REMOTE_MLFLOW_DIR} -> ${LOCAL_CACHE_DIR} ..."
sync_once

# Both the watcher loop and mlflow ui are started as background children
# with their PIDs captured explicitly, then cleaned up via a trap on exit
# (normal, Ctrl-C, or `kill` on this script's own PID) rather than via
# `exec`/foreground-process-group signal propagation -- there are two
# children to manage here (watcher + mlflow ui), and `exec`ing the last one
# would silently drop the trap (exec replaces this process, so bash never
# runs it) and leave the OTHER child (the watcher, if running) orphaned.
# Confirmed this class of bug bites in practice, for the simpler single-
# child case: see mlflow_tunnel_ulysses.sh's history, 2026-08-31.
if [[ "${WATCH_INTERVAL}" -gt 0 ]]; then
  (
    while true; do
      sleep "${WATCH_INTERVAL}"
      sync_once >/dev/null 2>&1 || true
    done
  ) &
  WATCHER_PID=$!
  echo "re-syncing every ${WATCH_INTERVAL}s in the background (pid ${WATCHER_PID}) while mlflow ui runs below"
else
  WATCHER_PID=""
fi

echo "open http://localhost:${LOCAL_PORT} once mlflow's \"Listening at\" line appears below."
echo "Ctrl-C stops mlflow ui (and the re-sync loop, if running)."
echo

# MLFLOW_ALLOW_FILE_STORE=true: `mlflow ui` refuses a file-store backend by
# default (mlflow 3.x deprecated it, see CLAUDE.md) unless this is set --
# the pipelines opt back in via `os.environ.setdefault(...)` at import time
# (pipelines/train.py etc.), but that's Python-process-local and doesn't
# apply here since this invokes the `mlflow` CLI directly.
MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui --backend-store-uri "file://${LOCAL_CACHE_DIR}" --host 127.0.0.1 --port "${LOCAL_PORT}" &
MLFLOW_PID=$!

trap 'kill "${MLFLOW_PID}" "${WATCHER_PID}" 2>/dev/null || true' EXIT INT TERM
wait "${MLFLOW_PID}"
