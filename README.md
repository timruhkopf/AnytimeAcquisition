# AnytimeAcquisition

Research repo for learning an **in-context acquisition function** for
Bayesian Optimization (BO), instead of hand-crafting one (EI/UCB/PI/ES) on
top of a surrogate.

## Abstract

The goal is a policy that directly optimizes the metric BO is actually
judged on -- **anytime log-incumbent AUC** -- rather than a surrogate +
hand-crafted acquisition heuristic. Two structural ideas drive the
approach: (1) a π0.5-style **ActionHead** that cross-attends into a
**frozen Prior-Fitted Network (PFN)**'s train-token KV cache, the same way
π0.5's action expert cross-attends into a frozen VLM backbone; (2) since we
control the training prior, we can search it directly with gradient
descent to generate expert-iteration (privileged-search) imitation
targets, without discrete-decision MCTS or RL (yet). See `docs/ROADMAP.md`
for the full design and rationale, `docs/MILESTONES.md` for the phased
checklist (M0-M7), and `docs/OPEN_QUESTIONS.md` for decisions that are
intentionally still open -- check that file before assuming a design
choice (benchmark suite, BO baseline library, medium-dim target, ...) has
already been made.

The repo was rebuilt from scratch on the `claude-init` branch; an earlier
implementation lives under `archive/` for reference and deliberate
porting-back, not as active code.

## Installation

Requires Python >=3.10 and [uv](https://docs.astral.sh/uv/).

`torch`/`botorch` are split into two conflicting extras (CPU vs. CUDA
12.4) rather than a plain dependency, since which build you need is a
per-machine choice `uv` can't infer on its own -- **always pass one of the
two explicitly**; a bare `uv sync` does not fail cleanly and does not skip
torch (see `pyproject.toml`'s own comment for the full story):

```bash
uv sync --extra cpu       # local dev / any CPU-only machine
uv sync --extra cu124     # a CUDA 12.4 GPU machine (confirmed correct on
                          # `ulysses`; NOT yet verified on LUIS's own GPU
                          # nodes -- see docs/OPEN_QUESTIONS.md #7)
```

Verify the install by running the test suite:

```bash
uv run pytest
```

## MLflow setup

Every pipeline logs to MLflow using the plain **file-store backend** (no
Postgres/SQLite server to coordinate, deliberate -- see `CLAUDE.md`) --
runs land under `mlruns/` at the repo root by default; set `AA_MLFLOW_DIR`
to point at shared/network storage instead (e.g. for parallel SLURM jobs).

Each pipeline logs to its **own MLflow experiment** (`anytimeacquisition-pfn-pretrain`,
`anytimeacquisition-exit-train`, `anytimeacquisition-action-head-posterior-distill`,
`anytimeacquisition-explore-search-playground`) rather than one shared
bucket -- see each top-level `configs/*.yaml`'s own `callbacks.mlflow.experiment_name`.
Metrics are namespaced with `/` (e.g. `train/nll`, `eval/mse`,
`generalize_real/beta_nll`) so MLflow's own chart view groups them by
prefix instead of listing everything flat.

Viewing the dashboard depends on where the run happened -- three separate
compute environments, each with its own `mlruns/`, not one shared "the
cluster":

- **Local** (this/your own machine): point `mlflow ui` straight at the
  tracking dir, e.g. `mlflow ui --backend-store-uri file://$(pwd)/mlruns`
  (or wherever `AA_MLFLOW_DIR` points), then open http://localhost:5000.
- **`ulysses`** (a personal, single-GPU machine used as a remote GPU
  interpreter, not a job scheduler): `scripts/mlflow_tunnel_ulysses.sh` --
  live SSH tunnel to a remote `mlflow ui`, fine since it's single-tenant.
- **LUIS** (the actual SLURM cluster, shared login node):
  `scripts/mlflow_sync_luis.sh` -- rsyncs `mlruns/` down over the dedicated
  transfer host, then views the synced copy purely locally; nothing
  long-running gets started on the shared login node.

## Configuration (Hydra)

Config groups live under `configs/`, one directory per group, mirroring
`src/anytimeacquisition/`'s own subpackages (`priors/`, `models/surrogates/`,
`trainer/`, `callbacks/`, ...). There are **two separate top-level
compositions**, not one shared config, since PFN pretraining doesn't touch
the EXIT/policy loop's `benchmarks`/`trainer` at all:

- `configs/train_pfn.yaml` -- PFN pretraining (`pipelines/train_pfn.py`,
  M2) -- the pipeline that's actually real and runnable today.
- `configs/config.yaml` -- the EXIT/policy-training loop (`pipelines/train.py`),
  still wired to a `DummyTrainer` placeholder pending M5/M6.

Named, reproducible run configs live under `configs/experiment/` (Hydra's
`# @package _global_` pattern), selected via `experiment=<name>` (not
`+experiment=<name>` -- it's already an optional default). `configs/pfn_checkpoint/`
is a third group worth knowing: a self-describing pointer to one specific
trained PFN checkpoint (path + architecture + provenance), so pipelines
that load a frozen PFN (`action_head_posterior_distill.py`,
`explore_search_playground.py`) reference a named entry instead of
duplicating a raw path + dimensionality by hand.

Start with `CLAUDE.md` for the full repo layout and working conventions,
and `docs/OPEN_QUESTIONS.md` before assuming any of the still-open design
questions above have been settled.

## Usage

Train a PFN (M2) -- a small, fast smoke run, a few minutes on CPU:

```bash
uv run python -m anytimeacquisition.pipelines.train_pfn experiment=pfn_smoke_xdim2
```

This trains against `priors/bnn.py`'s synthetic BNN prior and checkpoints
to `models/pfn_smoke_xdim2.pt` (gitignored, regenerable). See
`docs/milestones/M2.md` for what a real (non-smoke) training run should
look like, and `configs/experiment/` for the other named configs
(`pfn_smoke_xdim1`, `pfn_variable_xdim_smoke`, `pfn_ulysses_real`, ...).

Every component also has an interactive demo -- e.g.:

```bash
uv run python -m anytimeacquisition.models.pfn
uv run python -m anytimeacquisition.priors.bnn
```

See `CLAUDE.md`'s Commands section for the rest (local sweeps, SLURM
dispatch via `scripts/submit.sh`, ActionHead posterior-distillation
diagnostic, explore-search playground).
