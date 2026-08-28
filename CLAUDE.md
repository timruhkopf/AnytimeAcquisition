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
```

- `torch` is split into two conflicting `uv` extras (`cpu`/`cu126`), not a
  plain dependency — **always pass `--extra cpu` or `--extra cu126`**, a
  bare `uv sync` does not fail cleanly (see `pyproject.toml`'s comment for
  why). `cu126` is a guess for the SLURM cluster's CUDA version, unverified
  — check with `nvidia-smi` before trusting it (`docs/OPEN_QUESTIONS.md` #7).
- `scripts/submit.sh`/`scripts/slurm/train.sbatch` take the pipeline module
  as their first argument now (used to hardcode `pipelines.train`) and
  activate a pre-built `.venv` (`source .venv/bin/activate`), not a conda
  env — sync the venv once on the cluster's login node
  (`uv sync --extra cu126`) before submitting, don't rely on the job itself
  to provision anything (no `uv run` inside the sbatch script — compute
  nodes often have no network access).

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
  `./mlruns`, override with the `AA_MLFLOW_DIR` env var to point at shared
  cluster storage for a run to be visible across nodes.
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
