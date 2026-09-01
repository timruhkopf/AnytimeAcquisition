#!/bin/bash
# Stream ulysses' MLflow file-store dashboard onto your local browser over
# an SSH tunnel.
#
# `ulysses` is a personal, single-GPU office machine (SSH alias `ulysses`,
# jump-hosted through `ai-gateway`/ssh2.ai.uni-hannover.de) used as a
# PyCharm remote interpreter for small-scale/sanity-check GPU runs -- it is
# NOT the SLURM cluster (that's LUIS, see `scripts/mlflow_sync_luis.sh`
# instead) and not shared with anyone else, so running a live `mlflow ui`
# process on it for the duration of this script is fine.
#
# How it works: MLflow's file-store backend (see CLAUDE.md's "MLflow uses
# the file-store backend" convention) needs no server running during
# training -- `mlflow ui` just reads the `mlruns/` dir straight off disk
# whenever you actually want to look at the dashboard. This script SSHes to
# ulysses, starts `mlflow ui` bound to 127.0.0.1 (not exposed on the
# network) as the SSH session's own remote command, and forwards a local
# port to it in that same `ssh` invocation. Ctrl-C locally tears down the
# tunnel AND kills the remote `mlflow ui` process together (it dies with
# the SSH session) -- nothing left running afterward.
#
# Usage:
#   scripts/mlflow_tunnel_ulysses.sh                        # defaults below
#   scripts/mlflow_tunnel_ulysses.sh 5001                    # different local port
#   AA_REMOTE_MLFLOW_DIR=/some/other/mlruns scripts/mlflow_tunnel_ulysses.sh
# Then open http://localhost:5000 (or your chosen port) locally, once the
# "Listening at: ..." line appears below.
#
# Defaults assume:
#   - the `ulysses` SSH config alias is already set up locally (see the
#     repo-root `ulysses` file -- gitignored, untracked; login notes live
#     there, not here). Different alias? Override AA_REMOTE_HOST.
#   - the repo cloned at ~/PycharmProjects/AnytimeAcquisition on ulysses
#     (confirmed 2026-08-31 via `ssh ulysses find` -- PyCharm's remote
#     interpreter mirrors the local project path, NOT ~/AnytimeAcquisition;
#     an earlier version of this script guessed wrong and pointed at an
#     empty mlruns/, which is why the dashboard showed only the
#     auto-created "Default" experiment with no runs), with a `.venv`
#     already `uv sync --extra cu124`'d there (see CLAUDE.md's commands).
#     This script does NOT sync/install anything for you. Different clone
#     path? Override AA_REMOTE_REPO_DIR.
#   - MLflow's tracking dir is the default `${aa_root:}/mlruns` inside that
#     repo clone, i.e. AA_MLFLOW_DIR was NOT set for runs launched on
#     ulysses. If it was, set AA_REMOTE_MLFLOW_DIR to that same path.
#
# A THIRD gotcha in this family (2026-09-01, cost real debugging time --
# dashboard showed only an empty "Default" experiment even right after a
# fresh run, with no wrong-path or orphaned-process explanation): PyCharm's
# SSH *Python interpreter* (the thing that actually runs e.g. train_pfn.py
# when using ulysses as a remote interpreter) auto-uploads the project to
# its OWN ephemeral staging dir, /tmp/pycharm_project_<hash> -- a totally
# separate sync mechanism from the "Deployment" config
# (.idea/deployment.xml) that maps REMOTE_REPO_DIR above and excludes
# mlruns/models/outputs/multirun/archive/.venv from upload. The interpreter
# staging copy has NO such exclusions, and since `${aa_root:}` is
# `Path(__file__)`-anchored, every interpreter-launched run's real mlruns/
# lands under that /tmp path, not REMOTE_REPO_DIR -- even though the
# interpreter's own venv binary is genuinely at REMOTE_REPO_DIR/.venv,
# which is what makes this so easy to miss. Confirm which one you're
# hitting via `ssh ulysses ps aux | grep pipelines` -- if the running
# script's own path starts with /tmp/pycharm_project_, point this script's
# AA_REMOTE_MLFLOW_DIR at that dir's mlruns/ instead (the <hash> suffix
# regenerates whenever the interpreter is recreated, so don't hardcode it
# here). The durable fix is in PyCharm's interpreter settings (a sync-
# folders / path-mapping option to reuse the Deployment-mapped path instead
# of its own temp staging dir), not this script.
set -euo pipefail

REMOTE_HOST="${AA_REMOTE_HOST:-ulysses}"
REMOTE_REPO_DIR="${AA_REMOTE_REPO_DIR:-\$HOME/PycharmProjects/AnytimeAcquisition}"
REMOTE_MLFLOW_DIR="${AA_REMOTE_MLFLOW_DIR:-${REMOTE_REPO_DIR}/mlruns}"
REMOTE_PORT="${AA_REMOTE_MLFLOW_PORT:-5000}"
LOCAL_PORT="${1:-5000}"

# REMOTE_REPO_DIR/REMOTE_MLFLOW_DIR may literally contain an unexpanded
# `$HOME` (see the default above) -- deliberate: it must expand on the
# REMOTE end, once the remote shell parses REMOTE_CMD below, not here.
# MLFLOW_ALLOW_FILE_STORE=true: `mlflow ui` refuses a file-store backend by
# default (mlflow 3.x deprecated it, see CLAUDE.md) unless this is set --
# the pipelines opt back in via `os.environ.setdefault(...)` at import time
# (pipelines/train.py etc.), but that's Python-process-local and doesn't
# apply here since this invokes the `mlflow` CLI directly, not through one
# of those modules.
REMOTE_CMD="cd ${REMOTE_REPO_DIR} && source .venv/bin/activate && \
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri file://${REMOTE_MLFLOW_DIR} --host 127.0.0.1 --port ${REMOTE_PORT}"

echo "tunneling localhost:${LOCAL_PORT} -> ${REMOTE_HOST}:${REMOTE_PORT} (${REMOTE_MLFLOW_DIR})"
echo "open http://localhost:${LOCAL_PORT} once mlflow's \"Listening at\" line appears below."
echo "Ctrl-C stops both the tunnel and the remote mlflow ui process."
echo

# exec, not a plain call: replaces this script's process with ssh itself
# (same PID) rather than running it as a child -- otherwise killing the
# script (as opposed to an interactive Ctrl-C, which signals the whole
# foreground process group) leaves ssh and the tunnel running, orphaned.
#
# -tt: force a pseudo-terminal for the remote command even though this
# isn't an interactive session. Without a pty, ssh closing its connection
# doesn't reliably SIGHUP the remote command -- confirmed this bites in
# practice (2026-08-31): even after killing the local `ssh` process
# directly, `mlflow ui` kept running on ulysses' port 5000, still bound to
# the *old* wrong REMOTE_REPO_DIR default from before that was fixed too,
# which is why the dashboard showed an empty "Default" experiment instead
# of the real one. A pty ties the remote process's session to the
# connection properly, so closing the connection reliably tears it down.
exec ssh -tt -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "${REMOTE_HOST}" "${REMOTE_CMD}"
