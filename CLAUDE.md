# AnytimeAcquisition

Research repo for anytime acquisition functions (Bayesian optimization). The
repo was rebuilt from scratch on `claude-init` — the previous implementation
lives under `archive/` for reference and deliberate porting-back, not as
active code.

See `docs/ROADMAP.md` for the phased plan, `docs/MILESTONES.md` for the
current checklist, and `docs/OPEN_QUESTIONS.md` for decisions that are
intentionally still open — check that file before assuming a design choice
(benchmark suite, baseline BO library, cluster partition, etc.) has been made.

## Layout

- `src/anytimeacquisition/` — the package, one subpackage per Hydra config group:
  - `benchmarks/` — optimization problems / tasks
  - `priors/` — environments/priors the policy trains against
  - `models/surrogates/`, `models/acquisition/`, `models/baselines/` (e.g. GP + EI/UCB/PI/ES)
  - `trainer/` — training loops
  - `callbacks/` — e.g. MLflow logging
  - `deployment/` — run provenance, cluster-facing helpers
  - `pipelines/` — Hydra entry points (`train.py`, more to come)
- `configs/` — Hydra configs, one directory per group above, mirroring `src/`.
  `configs/config.yaml` is the top-level composition.
- `tests/` — pytest, mirrors `src/` structure as it grows.
- `notebooks/` — showcase/exploratory notebooks (e.g. `bnn_prior_showcase.ipynb`).
  Executed with outputs committed so they're viewable without rerunning;
  re-execute (`jupyter nbconvert --to notebook --execute --inplace <path>`)
  after changing the component they showcase, don't let them go stale.
- `scripts/` — `submit.sh` (SLURM dispatch) and `slurm/train.sbatch`.
- `archive/` — prior implementation (`src/`, `tests/`, `main.py`, `README.md`).
  Do not import from here; port specific pieces into `src/` deliberately.

## Commands

```
uv sync --extra cpu                  # install deps -- ALWAYS pass --extra, see pyproject.toml
uv run pytest                        # run tests
uv run python -m anytimeacquisition.pipelines.train [overrides...]
uv run python -m anytimeacquisition.pipelines.train -m seed=0,1,2,3  # local sweep
uv run python -m anytimeacquisition.pipelines.train_pfn experiment=pfn_smoke_xdim2  # M2: train a PFN
scripts/submit.sh <module> [overrides...]     # sbatch a single SLURM run
scripts/submit.sh <module> -m seed=0,1,2,3    # SLURM sweep via hydra-submitit-launcher
# e.g. scripts/submit.sh anytimeacquisition.pipelines.train_pfn experiment=pfn_ulysses_real
scripts/mlflow_tunnel_ulysses.sh [local_port] # live SSH tunnel to ulysses' MLflow dashboard
scripts/mlflow_sync_luis.sh [local_port]      # rsync LUIS's mlruns/ down, then view locally
```

Three separate compute environments, each with its own MLflow data — not one
"the cluster": **local** (this laptop, CPU-only, debugging), **ulysses**
(personal single-GPU office machine, SSH alias `ulysses`, used as a PyCharm
remote interpreter for small-scale/sanity-check GPU runs — not a job
scheduler), and **LUIS** (the actual SLURM cluster, SSH alias `luis`, shared
login node — heavy training runs go here via `scripts/submit.sh`). See
`scripts/mlflow_tunnel_ulysses.sh` / `scripts/mlflow_sync_luis.sh` for why
they use different mechanisms (live tunnel vs. sync-then-view) — the login
node's shared, so nothing long-running gets started there.

- `torch` is split into two conflicting `uv` extras (`cpu`/`cu124`), not a
  plain dependency — **always pass `--extra cpu` or `--extra cu124`**, a
  bare `uv sync` does not fail cleanly (see `pyproject.toml`'s comment for
  why). `cu124` is confirmed correct on `ulysses` (`nvidia-smi`, verified
  2026-08-31) but **not yet checked on LUIS** — its GPU nodes are a
  separate machine with a possibly-different driver, don't assume it's the
  same without checking (`docs/OPEN_QUESTIONS.md` #7).
- `scripts/submit.sh`/`scripts/slurm/train.sbatch` target LUIS (the actual
  SLURM cluster, not `ulysses` — see `docs/OPEN_QUESTIONS.md` #4), take the
  pipeline module as their first argument now (used to hardcode
  `pipelines.train`), and activate a pre-built `.venv` (`source
  .venv/bin/activate`), not a conda env — sync the venv once on LUIS's
  login node (`uv sync --extra cu124`, or `cpu` if that turns out wrong for
  LUIS's GPU nodes) before submitting, don't rely on the job itself to
  provision anything (no `uv run` inside the sbatch script — compute nodes
  often have no network access).

- `configs/train_pfn.yaml` is a **separate** top-level Hydra composition
  from `configs/config.yaml` — PFN pretraining (M2) doesn't touch
  `benchmarks`/the EXIT `trainer`. Named, reproducible run configs for it
  live under `configs/experiment/` (Hydra's `# @package _global_`
  experiment-config pattern) and are selected via `experiment=<name>`, not
  `+experiment=<name>` (it's already an optional default, not a group
  being added fresh).

## Conventions

- **Every tracked run is tied to a clean git commit.** `anytimeacquisition.deployment.provenance.record_provenance`
  refuses to run against a dirty working tree (raises `DirtyRepoError`) unless
  `allow_dirty=true` is passed — that's for local debugging only, never for a
  run you intend to keep. It also captures the Hydra overrides used, and both
  get logged as MLflow tags (`git_commit`, `git_dirty`, `hydra_overrides`).
- **MLflow uses the file-store backend, not SQLite or Postgres** — deliberate,
  because there's no DB server to coordinate across parallel SLURM jobs
  (mlflow 3.x deprecated the plain file store; `train.py` sets
  `MLFLOW_ALLOW_FILE_STORE=true` to opt back in). Tracking dir defaults to
  `<repo_root>/mlruns`, override with the `AA_MLFLOW_DIR` env var to point at
  shared cluster storage for a run to be visible across nodes.
- **`outputs/`, `multirun/`, `mlruns/`, and `models/` are top-level,
  gitignored siblings of `src/`** — `src/anytimeacquisition/utils/paths.py`
  registers a custom `aa_root` OmegaConf resolver (`${aa_root:}` in
  configs/*.yaml) from `PROJECT_ROOT`, a `Path(__file__)`-anchored constant,
  not cwd-derived; `anytimeacquisition/__init__.py` imports `utils/paths.py`
  unconditionally so the resolver is always registered before any pipeline's
  `hydra.main`-decorated `main()` runs, with no per-pipeline setup needed.
  `configs/*.yaml` interpolate `${aa_root:}` for `hydra.run.dir`/
  `hydra.sweep.dir`, the MLflow tracking URI, and `checkpoint_path` — not
  Hydra's own `${hydra:runtime.cwd}`, which silently re-nests all of these
  under wherever a pipeline happens to be launched from (e.g. an IDE run
  config that defaults to the script's own directory) instead of the repo
  root. Non-Hydra call sites (demo `__main__` blocks, `pipelines/
  train_pfn.py`'s plain `train_pfn()` function) import
  `anytimeacquisition.utils.paths.PROJECT_ROOT`/`CHECKPOINT_DIR` directly
  instead. New configs referencing these dirs should use `${aa_root:}`, not
  reintroduce `${hydra:runtime.cwd}`.
- **New components get a Hydra config group**, not hardcoded wiring — add
  `src/anytimeacquisition/<group>/<name>.py` + `configs/<group>/<name>.yaml`
  with a matching `_target_`, and reference it from `configs/config.yaml`'s
  `defaults` or via override.
- **Everything gets a pytest test.** `tests/test_hydra_config.py` shows the
  pattern for testing config composition + instantiation; follow it for new
  components rather than only testing them manually.
- **Every component worth interacting with directly gets an `if __name__ ==
  "__main__":` demo**, not just pytest coverage — a way to actually run it
  and see what it does (load a checkpoint and show a search converging, run
  a forward pass and print/plot shapes and a sanity metric, etc.), skipped
  only when a component genuinely has nothing to demo (e.g. a plain config
  dataclass). This matters most for the research-critical pieces: showing
  the explore-branch entropy-gradient search actually reduces predictive
  entropy on a trained PFN checkpoint, showing the exploit-branch search
  actually finds better labels, and showing the ActionHead's cross-attention
  into the frozen PFN's KV cache is actually carrying signal (not just
  shape-checking, e.g. a "blind" ablation comparison).
- Don't silently make architecture/library decisions on the open items in
  `docs/OPEN_QUESTIONS.md` (benchmark suite, BO library for baselines, cluster
  resource specs, etc.) — ask first.
