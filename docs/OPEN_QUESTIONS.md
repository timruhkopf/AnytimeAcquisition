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

3. ~~Baseline BO library.~~ **Resolved 2026-08-28: BoTorch.** GP + EI/PI/ES
   (Max-value Entropy Search) landed in
   `models/baselines/gp_acquisition.py` + `configs/models/baselines/{ei,pi,es}.yaml`,
   sanity-checked (M6.md's required check) against random search on the same
   `BNNPrior` instances — all three beat it. UCB (also listed in M6's
   original scope) not built yet, trivial to add the same way if wanted.
   `models/surrogates/` (the PFN) does not share machinery with these —
   deliberately independent, since the whole comparison is "learned
   in-context acquisition vs. classical GP + hand-crafted acquisition,"
   sharing a surrogate would blur that.

4. **SLURM cluster specifics.** `scripts/slurm/train.sbatch` and
   `configs/deployment/slurm.yaml` have TODOs for partition, account, QOS,
   time limits, and GPU requests — need real values for the actual SLURM
   cluster, LUIS (`luis` / `login.cluster.uni-hannover.de` per the user's
   own SSH config; a dedicated `luis-transfer` /
   `transfer.cluster.uni-hannover.de` host also exists for data transfer,
   not the login node). **Not `ulysses`** — that's a separate, personal
   single-GPU office machine (SSH alias `ulysses`, jump-hosted through
   `ai-gateway`/`ssh2.ai.uni-hannover.de`) used as a PyCharm remote
   interpreter for small-scale/sanity-check GPU runs, not a job scheduler —
   earlier notes here (and the repo-root `ulysses` file's name) conflated
   the two; corrected 2026-08-31.

5. ~~uv vs. conda on the cluster.~~ **Resolved 2026-08-28: uv.**
   `scripts/slurm/train.sbatch` and `scripts/submit.sh` now `source
   .venv/bin/activate` (a `.venv` built ahead of time via `uv sync --extra
   cu124` on the login node) rather than activating the `anytimeacquisition`
   conda env. The conda env referenced in the `ulysses` notes is no longer
   what these scripts use — if anything still depends on it, that's now
   stale.

6. ~~`AA_MLFLOW_DIR` shared path.~~ **Resolved 2026-09-01: redirect
   `AA_PROJECT_ROOT` to `$BIGWORK/AnytimeAcquisition`.**
   Per LUIS's storage docs (docs.cluster.uni-hannover.de/doku.php/guide/storage_systems):
   `$HOME` is explicitly "do not put data here that you use in compute
   jobs" (slow NFS, 10GB/12GB quota); `$PROJECT` (group storage, 10TB) is
   only mounted on login/transfer nodes, **not compute nodes**, so parallel
   jobs can't write there mid-run at all; `$BIGWORK` (InfiniBand, all
   nodes including compute, 100GB/1TB per-user quota, no backup — treat as
   scratch) is the only one both fast and reachable from every job. Rather
   than redirecting `AA_MLFLOW_DIR` alone, `scripts/submit.sh`/
   `scripts/slurm/train.sbatch` redirect `AA_PROJECT_ROOT` instead — the
   `aa_root` resolver (`src/anytimeacquisition/utils/paths.py`) backs
   `hydra.run.dir`/`hydra.sweep.dir` (and therefore hydra-submitit-launcher's
   own `.submitit/` job-log dir, which nests under `hydra.sweep.dir`),
   `checkpoint_path`, *and* MLflow's tracking dir, so redirecting only the
   latter would've left Hydra/submitit logs and checkpoints still
   defaulting under `$HOME`. `AA_MLFLOW_DIR` remains available on top of
   this for pointing mlflow data somewhere different from the rest (e.g.
   `$PROJECT` for archival) — see `configs/callbacks/mlflow.yaml`.
   `scripts/mlflow_sync_luis.sh` resolves the same `$BIGWORK` path via a
   login-shell SSH round-trip rather than assuming it's set in a bare
   non-interactive SSH command's environment. Since `$BIGWORK` has no
   backup, copy finished runs to `$PROJECT` manually if you want them
   retained long-term — nothing here automates that yet.

7. ~~Python/torch version pinning, and CUDA torch for `ulysses`.~~
   **Partially resolved 2026-08-31: `cu124` confirmed correct on
   `ulysses`** (`nvidia-smi` shows driver supports CUDA 12.4) — the extra
   was renamed from the earlier `cu126` guess to `cu124` in
   `pyproject.toml`, fully wired (`[tool.uv.sources]`/`[[tool.uv.index]]`
   pointing at `download.pytorch.org/whl/cu124`), `botorch` included.
   **Still open for LUIS**: LUIS's actual GPU nodes are a separate machine
   from `ulysses` (see item 4's 2026-08-31 correction) with a
   possibly-different driver — `cu124` is a reasonable starting guess since
   it matched `ulysses`, but not verified there. Run `nvidia-smi` on a LUIS
   GPU node (or check its docs) before trusting it for a real run; if it
   needs a different CUDA version, add another `[[tool.uv.index]]` entry +
   extra (same pattern as `cpu`/`cu124`) rather than changing the existing
   one, so both stay available in one lockfile.
   `pyproject.toml` currently allows `torch>=2.13.0` and `python>=3.10` as
   a floor — fine. Also confirmed empirically: **always pass an `--extra`
   flag** — a bare `uv sync` does not fail cleanly and does not skip torch,
   it silently falls through to the unconstrained default GPU wheel anyway
   (`tabpfn` also depends on torch, untied to either of our extras) — see
   `pyproject.toml`'s comment.
   Separately: `trainer/pfn_trainer.py`'s `mixed_precision` (AMP) support
   still needs validating on real GPU hardware — implemented and CPU-safe,
   but never actually exercised on CUDA (not blocked on the LUIS driver
   question above — could be checked on `ulysses` now that its CUDA
   version is confirmed).

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

