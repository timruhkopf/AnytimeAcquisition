"""Standalone experimentation + regret-validation sandbox for M5's explore
branch (`search/explore.py`) — built specifically to answer the question
that matters and that nothing else in this repo checks directly: **does a
correction found by `explore_search` actually help reduce regret, or does
it merely reduce the weighted-NLL proxy we optimize?** Full design history:
`docs/log/2026-08-28-explore-search-input-optimization-and-teacher-forcing.md`.

`explore_search` differentiates a proxy (weighted NLL of privileged
`y_int_true` under the PPD, see that module's docstring for why NLL and not
entropy). A proxy improving is not the same claim as "this reduces regret"
-- this pipeline closes that gap using privileged information the search
itself doesn't have access to as an optimization target (it only uses
`y_int_true` for *weighting*, never as a direct regret computation), namely
the **greedy regret** of the model's own point estimate:

  `greedy_regret(context) = y_int_true[argmin_i predicted_mean_i] - y_int_true.min()`

-- i.e. "if a downstream policy greedily committed to whichever `x_int`
point the model's own predicted mean currently favors, how far off would
that choice be from the actual best option available?" This is exactly the
literal, decision-theoretic regret criterion discussed as a *more direct*
alternative to NLL (a softmin-relaxed differentiable version of it), but
used here purely as a **non-differentiable, post-hoc diagnostic** — no
`argmin`-through-gradients anywhere, since nothing here needs to
differentiate through it. It's the right tool for validation even though
it was the wrong tool (per 2026-08-28 discussion) for the optimization
objective itself.

For every explore-labeled (instance, step) pair across `n_episodes` fresh
rollouts, this pipeline computes `greedy_regret` before and after adding
`explore_search`'s correction, and reports:
  - mean/median regret reduction, and the fraction of examples where regret
    actually got worse (not swept under an aggregate mean)
  - the correlation between weighted-NLL improvement (what's optimized) and
    true regret reduction (what we actually care about) -- the single
    number that most directly answers "is NLL a good proxy here"

Deliberately no Hydra config group for a new modeling component here (same
scoping call as `action_head_posterior_distill.py`): this is a diagnostic
tool, not a permanent pipeline other code depends on.
"""
import logging
import os
from pathlib import Path
from typing import Callable

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import hydra
import mlflow
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from anytimeacquisition.deployment.provenance import record_provenance
from anytimeacquisition.models.bar_distribution import BarDistribution
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.pipelines.train_pfn import load_pfn_checkpoint
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.search.explore import improvement_weights
from anytimeacquisition.trainer.exit_rollout import build_explore_buffer, random_policy, rollout_episode
from anytimeacquisition.utils.flatten import flatten
from anytimeacquisition.utils.paths import CHECKPOINT_DIR

log = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "pfn_smoke_xdim1.pt"


@torch.no_grad()
def greedy_regret(
    pfn: PFN, bar_dist: BarDistribution, x_context: torch.Tensor, y_context: torch.Tensor,
    x_int: torch.Tensor, y_int_true: torch.Tensor,
) -> torch.Tensor:
    """True regret of the model's own greedy pick among `x_int` (argmin of
    predicted mean), given `x_context`/`y_context`. Non-differentiable,
    diagnostic only -- see module docstring for why that's fine here.
    x_context/y_context: [B,Nt,x_dim]/[B,Nt]  x_int/y_int_true: [B,N_int,x_dim]/[B,N_int]
    -> [B]."""
    predicted_means = bar_dist.mean(pfn(x_context, y_context, x_int))  # [B, N_int]
    greedy_idx = predicted_means.argmin(dim=1)  # [B]
    B = x_context.shape[0]
    picked_y = y_int_true[torch.arange(B), greedy_idx]
    best_y = y_int_true.min(dim=1).values
    return picked_y - best_y


def _pearson_corr(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return float("nan")
    a_t, b_t = torch.tensor(a), torch.tensor(b)
    a_t, b_t = a_t - a_t.mean(), b_t - b_t.mean()
    denom = a_t.norm() * b_t.norm()
    return (a_t @ b_t / denom).item() if denom > 0 else float("nan")


def run_explore_playground(
    pfn: PFN, bar_dist: BarDistribution, x_dim: int, prior_batch_size: int, seed: int,
    n_episodes: int, n_init: int, n_steps: int,
    n_sobol: int, n_random: int, n_basin_restarts: int,
    explore_n_restarts: int, explore_n_steps: int, explore_lr: float,
    on_log: Callable[[int, dict], None] | None = None,
) -> dict:
    """Runs `n_episodes` fresh self-play rollouts (random policy, own fresh
    `BNNPrior` + `x_int`/`y_int_true` per episode), collects every explore
    branch correction found, and evaluates each one's effect on
    `greedy_regret` against its own pre-correction context. Returns the
    aggregate statistics plus every raw per-example measurement (so a
    caller can plot/inspect the distribution, not just the summary)."""
    torch.manual_seed(seed)

    regret_before, regret_after = [], []
    weighted_nll_before, weighted_nll_after = [], []

    for episode in range(n_episodes):
        prior = BNNPrior(batch_size=prior_batch_size, x_dim=x_dim, seed=seed + episode)
        rollout = rollout_episode(
            prior, n_init=n_init, n_steps=n_steps, policy_fn=random_policy,
            build_interesting_points_kwargs={
                "n_sobol": n_sobol, "n_random": n_random, "n_basin_restarts": n_basin_restarts,
            },
        )
        buffer = build_explore_buffer(
            prior, pfn, bar_dist, rollout, n_init,
            explore_search_kwargs={"n_restarts": explore_n_restarts, "n_steps": explore_n_steps, "lr": explore_lr},
        )
        x_int, y_int_true = rollout["x_int"], rollout["y_int_true"]

        for ex in buffer:
            x_ctx0, y_ctx0 = ex.x_context.unsqueeze(0), ex.y_context.unsqueeze(0)
            x_int_b = x_int[ex.instance_idx].unsqueeze(0)
            y_int_true_b = y_int_true[ex.instance_idx].unsqueeze(0)

            weights = improvement_weights(y_ctx0.min(dim=1).values, y_int_true_b)
            with torch.no_grad():
                nll_before = bar_dist(pfn(x_ctx0, y_ctx0, x_int_b), y_int_true_b)
            r_before = greedy_regret(pfn, bar_dist, x_ctx0, y_ctx0, x_int_b, y_int_true_b)

            # prior.evaluate requires its leading dim to exactly match
            # prior.B (each row is a distinct instance's own weights) -- pad
            # to the full batch, evaluate, then keep only this example's own
            # row. Wasteful (computes B-1 unused rows) but simple, and cheap
            # at this pipeline's scale.
            x_pad = torch.zeros(prior.B, 1, x_dim)
            x_pad[ex.instance_idx, 0] = ex.x_star
            with torch.no_grad():
                y_star_true = prior.evaluate(x_pad, noise=False)[ex.instance_idx].unsqueeze(0)
            x_aug = torch.cat([x_ctx0, ex.x_star.view(1, 1, -1)], dim=1)
            y_aug = torch.cat([y_ctx0, y_star_true], dim=1)
            r_after = greedy_regret(pfn, bar_dist, x_aug, y_aug, x_int_b, y_int_true_b)

            regret_before.append(r_before.item())
            regret_after.append(r_after.item())
            weighted_nll_before.append((weights * nll_before).sum().item())
            weighted_nll_after.append(ex.y_star.item())

        if on_log is not None:
            on_log(episode, {"buffer_size": len(buffer)})
        print(f"  episode {episode:3d}: {len(buffer)} explore corrections found "
              f"({len(regret_before)} total so far)")

    n = len(regret_before)
    if n == 0:
        raise RuntimeError(
            "no explore-branch corrections with signal were found across any episode -- "
            "increase n_episodes/n_steps, or check that x_int actually contains points below "
            "the incumbent (has_signal=False everywhere is a legitimate but unhelpful outcome here)"
        )
    regret_reduction = [b - a for b, a in zip(regret_before, regret_after)]
    nll_improvement = [b - a for b, a in zip(weighted_nll_before, weighted_nll_after)]

    result = {
        "n_examples": n,
        "mean_regret_before": sum(regret_before) / n,
        "mean_regret_after": sum(regret_after) / n,
        "mean_regret_reduction": sum(regret_reduction) / n,
        "frac_regret_improved": sum(r > 1e-6 for r in regret_reduction) / n,
        "frac_regret_worsened": sum(r < -1e-6 for r in regret_reduction) / n,
        "mean_weighted_nll_improvement": sum(nll_improvement) / n,
        "nll_improvement_vs_regret_reduction_corr": _pearson_corr(nll_improvement, regret_reduction),
        "regret_before": regret_before,
        "regret_after": regret_after,
        "weighted_nll_before": weighted_nll_before,
        "weighted_nll_after": weighted_nll_after,
    }
    return result


@hydra.main(config_path="../../../configs", config_name="explore_search_playground", version_base=None)
def main(cfg: DictConfig) -> dict:
    """Hydra entry point:
      uv run python -m anytimeacquisition.pipelines.explore_search_playground
    """
    overrides = HydraConfig.get().overrides.task
    provenance = record_provenance(list(overrides), allow_dirty=cfg.get("allow_dirty", False))

    mlflow.set_tracking_uri(cfg.callbacks.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.callbacks.mlflow.experiment_name)

    pfn, bar_dist, ckpt = load_pfn_checkpoint(cfg.checkpoint_path)
    print(f"loaded PFN checkpoint: {Path(cfg.checkpoint_path).name}, config={ckpt['config']}")
    if ckpt["config"]["max_x_dim"] != cfg.x_dim:
        raise ValueError(
            f"cfg.x_dim={cfg.x_dim} does not match the checkpoint's own max_x_dim={ckpt['config']['max_x_dim']} "
            f"({cfg.checkpoint_path}) -- override x_dim to match."
        )

    with mlflow.start_run():
        mlflow.set_tags(provenance.as_mlflow_tags())
        # Checkpoint lineage (train_pfn.py's main()) -- see the same tags in
        # action_head_posterior_distill.py's main() for why "unknown" rather
        # than omitting the tag for an older/non-Hydra-trained checkpoint.
        mlflow.set_tags({
            "pfn_mlflow_run_id": ckpt.get("mlflow_run_id") or "unknown",
            "pfn_git_commit": ckpt.get("git_commit") or "unknown",
        })
        mlflow.log_params(flatten(OmegaConf.to_container(cfg, resolve=True)))
        mlflow.log_params(flatten({"pfn_checkpoint": dict(ckpt["config"])}))

        result = run_explore_playground(
            pfn=pfn, bar_dist=bar_dist, x_dim=cfg.x_dim, prior_batch_size=cfg.prior_batch_size, seed=cfg.seed,
            n_episodes=cfg.n_episodes, n_init=cfg.n_init, n_steps=cfg.n_steps,
            n_sobol=cfg.interesting_points.n_sobol, n_random=cfg.interesting_points.n_random,
            n_basin_restarts=cfg.interesting_points.n_basin_restarts,
            explore_n_restarts=cfg.explore_search.n_restarts, explore_n_steps=cfg.explore_search.n_steps,
            explore_lr=cfg.explore_search.lr,
            on_log=lambda step, metrics: mlflow.log_metrics(metrics, step=step),
        )

        summary = {k: v for k, v in result.items() if not isinstance(v, list)}
        mlflow.log_metrics(summary)

        print("\n=== summary ===")
        for k, v in summary.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        log.info("run complete: %s", summary)
        return result


if __name__ == "__main__":
    main()
