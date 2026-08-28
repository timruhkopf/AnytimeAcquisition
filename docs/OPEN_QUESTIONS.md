# Open Questions

Decisions that are intentionally *not* made yet — flagged here instead of
picked silently. Resolve and delete/move to a changelog as they're settled.

1. **Evaluation benchmark.** Phases 5/6 evaluate on held-out instances of
   the same synthetic BNN prior (`benchmarks/` = more draws from `priors/`),
   or a separate real/established suite (HPOBench/JAHS-Bench/specific tasks)
   for out-of-distribution evaluation, or both (synthetic during EXIT
   development, real suite as a later held-out check)? Affects `benchmarks/`
   and how `models/baselines/` gets wired to it.

2. **Exact medium-dimensionality target.** "Medium, not low, not high" is
   settled (see `docs/ROADMAP.md`) but an actual `x_dim` range isn't pinned
   yet — needed to configure Phase 1's prior, Phase 2's PFN retraining, and
   Phase 6's baselines consistently.

3. **Baseline BO library.** For GP + EI/UCB/PI/ES baselines: BoTorch,
   GPyTorch, scikit-optimize, or something else? Affects `models/baselines/`
   and possibly `models/surrogates/` if the surrogate should share machinery
   with the baselines.

4. **SLURM cluster specifics.** `scripts/slurm/train.sbatch` and
   `configs/deployment/slurm.yaml` have TODOs for partition, account, QOS,
   time limits, and GPU requests — need real values for your cluster
   (`ulysses`/`ssh2.ai.uni-hannover.de` per the notes at repo root).

5. ~~uv vs. conda on the cluster.~~ **Resolved 2026-08-28: uv.**
   `scripts/slurm/train.sbatch` and `scripts/submit.sh` now `source
   .venv/bin/activate` (a `.venv` built ahead of time via `uv sync --extra
   cu126` on the login node) rather than activating the `anytimeacquisition`
   conda env. The conda env referenced in the `ulysses` notes is no longer
   what these scripts use — if anything still depends on it, that's now
   stale.

6. **`AA_MLFLOW_DIR` shared path.** What network-filesystem path should
   parallel SLURM jobs point MLflow's file store at so all runs land in one
   place?

7. **Python/torch version pinning, and CUDA torch for the cluster.**
   `pyproject.toml` currently allows `torch>=2.13.0` and `python>=3.10` —
   fine as a floor. As of 2026-08-28, torch is split into two conflicting
   `uv` extras (`cpu`/`cu126`, see `pyproject.toml`'s own comments) instead
   of one hardcoded index — `uv sync --extra cpu` locally (no working CUDA
   driver on this machine, and the default GPU wheel's bundled nvidia-*
   packages turned out to intermittently corrupt on download here),
   `uv sync --extra cu126` on Ulysses. **`cu126` is a guess, not verified**
   — run `nvidia-smi` on Ulysses to check the actual driver's max supported
   CUDA version before relying on it; if it needs a different one, add
   another `[[tool.uv.index]]` entry and extra (same pattern) rather than
   just changing the URL, so both variants stay available in one lockfile.
   Also confirmed empirically: **always pass an `--extra` flag** — a bare
   `uv sync` does not fail cleanly and does not skip torch, it silently
   falls through to the unconstrained default GPU wheel anyway (`tabpfn`
   also depends on torch, untied to either of our extras) — see
   `pyproject.toml`'s comment. Once cu126 (or whatever's correct) is
   confirmed: `trainer/pfn_trainer.py`'s `mixed_precision` (AMP) support
   needs validating on real GPU hardware too — implemented and CPU-safe,
   but never actually exercised on CUDA.

8. ~~ECDF-normalizing the prior's output vs. adaptive bar-distribution bin
   borders.~~ **Resolved 2026-08-28**: bounded `BarDistribution` on M1's
   existing `[0,1]`-normalized output (`src/anytimeacquisition/models/bar_distribution.py`),
   not PFNs4BO's unbounded `FullSupportBarDistribution` with adaptively-fit
   borders — chosen deliberately for M2's first pass (simpler, already-built
   on M1's ECDF machinery), not because the full-support approach was wrong.
   Revisit if the bounded head turns out limiting once M4/M5 build on it —
   see `docs/log/2026-08-28-m2-pfn-and-bar-distribution.md`.

9. **`.idea/` and `resources/`** (7.4MB, untracked) were left alone during
   the archive move since they weren't mentioned — gitignore them, add them,
   or something else?

