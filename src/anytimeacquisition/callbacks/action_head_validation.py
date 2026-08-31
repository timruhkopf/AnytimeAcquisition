"""ActionHead imitation-training validation callbacks (M5) --
`callbacks/dim_validation.py`'s factory-function pattern, applied to
`trainer.action_head_imitation_trainer.ActionHeadImitationTrainer`.

Every callback here rebuilds its held-out `BNNPrior`(s) fresh, from the
same fixed seed, at every single call -- not built once and reused like
`dim_validation.py`'s dedicated priors. `trainer.exit_rollout.rollout_episode`
always calls `prior.reset()` internally (there's no way to roll out without
one), so a prior built once and reused would still draw a fresh
architecture on every call anyway; rebuilding from the same seed instead
means every callback tick (and, for `build_auc_eval_callback`, every
policy compared within one tick) probes the *exact same* held-out
instance(s) -- same idea as `models/baselines/gp_acquisition.py`'s own
`__main__` demo (`torch.manual_seed` + fresh same-seeded `BNNPrior` before
each policy's rollout gives byte-identical underlying instances until the
policies' own choices diverge).
"""
from functools import partial
from typing import Any

import mlflow
import torch

from anytimeacquisition.callbacks.handler import Callback
from anytimeacquisition.metrics.inc_auc import incumbent_trajectory, log_incumbent_auc
from anytimeacquisition.models.action_head import action_head_policy_fn, beta_mode, build_rollout_aux_features
from anytimeacquisition.models.baselines.gp_acquisition import gp_acquisition_policy
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.trainer.exit_rollout import (
    build_exploit_buffer,
    build_explore_buffer,
    label_branches,
    random_policy,
    rollout_episode,
)


def _held_out_prior(x_dim: int, batch_size: int, seed: int, prior_kwargs: dict | None) -> BNNPrior:
    return BNNPrior(batch_size=batch_size, x_dim=x_dim, seed=seed, **(prior_kwargs or {}))


def build_auc_eval_callback(
    x_dim: int,
    n_init: int,
    n_steps: int,
    eval_batch_size: int = 8,
    eval_seed: int = 999,
    prior_kwargs: dict | None = None,
    ei_kwargs: dict | None = None,
    log_figure: bool = True,
    every_n_steps: int | None = None,
) -> Callback:
    """The project's north-star metric (`docs/ROADMAP.md`): mean
    `log_incumbent_auc` for the ActionHead's *current* policy
    (`action_head_policy_fn(trainer.action_head, trainer.pfn, n_steps,
    sample=False)`) vs. two baselines -- `random_policy` and the already-
    built GP+EI baseline (`models.baselines.gp_acquisition.gp_acquisition_policy`,
    already a drop-in `policy_fn`) -- all three rolled out against fresh
    `BNNPrior(seed=eval_seed, ...)` instances (same seed for all three, so
    they're compared on the literal same underlying BNN draws, not just the
    same distribution -- see module docstring).

    Reports `auc/action_head`, `auc/random`, `auc/ei` (metric-type-first,
    one chart groups all three) at the *outer* training-step cadence. Also
    logs a small comparison figure (mean log-incumbent per rollout step,
    one line per policy) via `mlflow.log_figure` at the same cadence --
    that's a genuinely different x-axis (rollout step, not training step)
    that doesn't belong forced into a scalar `mlflow.log_metric` series (see
    the plan addendum this was built from) -- an MLflow *artifact*, so it's
    a no-op (not an error) when there's no active run, e.g. under test.

    Deliberately ONE fixed `x_dim` (the PFN's own `max_x_dim`), not a swept
    list like `dim_validation.py` -- avoids per-dimension metric-name bloat
    for a callback whose whole point is the action_head/random/EI
    comparison, not a dimensionality sweep.
    """
    ei_kwargs = ei_kwargs or {}

    def probe(step: int, trainer: Any) -> dict:
        policies = {
            "action_head": action_head_policy_fn(trainer.action_head, trainer.pfn, n_steps, sample=False),
            "random": random_policy,
            "ei": partial(gp_acquisition_policy, acquisition="EI", **ei_kwargs),
        }
        curves, metrics = {}, {}
        for name, policy_fn in policies.items():
            prior = _held_out_prior(x_dim, eval_batch_size, eval_seed, prior_kwargs)
            rollout = rollout_episode(prior, n_init, n_steps, policy_fn=policy_fn)
            y = rollout["y_context"][:, n_init:]  # policy-chosen steps only, matching n_steps
            metrics[f"auc/{name}"] = log_incumbent_auc(y).mean().item()
            curves[name] = torch.log(incumbent_trajectory(y).clamp_min(1e-12)).mean(dim=0)  # [n_steps]

        if log_figure and mlflow.active_run() is not None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            for name, curve in curves.items():
                ax.plot(range(n_steps), curve.tolist(), label=name)
            ax.set_xlabel("rollout step")
            ax.set_ylabel("mean log(incumbent)")
            ax.set_title(f"held-out incumbent curves at training step {step}")
            ax.legend(fontsize="small")
            ax.grid(True, alpha=0.3)
            mlflow.log_figure(fig, f"incumbent_curves/step_{step}.png")
            plt.close(fig)

        return metrics

    return Callback(name="", fn=probe, every_n_steps=every_n_steps)


def _held_out_l1(trainer: Any, n_init: int, n_steps: int, eval_seed: int, batch_size: int,
                  prior_kwargs: dict | None, blind: bool) -> dict:
    """Shared machinery for `build_held_out_target_l1_callback` and
    `build_blind_ablation_callback` -- a fresh held-out `random_policy`
    rollout, the SAME oracle-target machinery training uses
    (`build_exploit_buffer`/`build_explore_buffer`), mean L1 between
    `beta_mode(action_head(...))` and the oracle's own `x_star`, per branch.
    """
    prior = _held_out_prior(trainer.prior.d, batch_size, eval_seed, prior_kwargs)
    build_ip_kwargs = trainer.build_interesting_points_kwargs if "explore" in trainer.branches else None
    rollout = rollout_episode(prior, n_init, n_steps, policy_fn=random_policy, build_interesting_points_kwargs=build_ip_kwargs)

    examples = []
    if "exploit" in trainer.branches:
        examples += build_exploit_buffer(prior, rollout, n_init, trainer.exploit_search_kwargs)
    if "explore" in trainer.branches:
        examples += build_explore_buffer(prior, trainer.pfn, trainer.bar_dist, rollout, n_init, trainer.explore_search_kwargs)

    by_step: dict[int, list] = {}
    for ex in examples:
        by_step.setdefault(ex.step, []).append(ex)

    l1_sums = {"exploit": 0.0, "explore": 0.0}
    l1_counts = {"exploit": 0, "explore": 0}
    for step, step_examples in by_step.items():
        x_context = torch.stack([ex.x_context for ex in step_examples])
        y_context = torch.stack([ex.y_context for ex in step_examples])
        x_star = torch.stack([ex.x_star for ex in step_examples])
        aux = build_rollout_aux_features(rollout, step, n_steps)
        idx = torch.tensor([ex.instance_idx for ex in step_examples])
        aux = {k: v[idx] for k, v in aux.items()}
        with torch.no_grad():
            out = trainer.action_head(trainer.pfn, x_context, y_context, aux, blind=blind)
            mode = beta_mode(out["alpha"], out["beta"])
        l1_per_example = (mode - x_star).abs().sum(dim=-1)
        for i, ex in enumerate(step_examples):
            l1_sums[ex.branch] += l1_per_example[i].item()
            l1_counts[ex.branch] += 1

    return {
        branch: (l1_sums[branch] / l1_counts[branch] if l1_counts[branch] else float("nan"))
        for branch in ("exploit", "explore") if branch in trainer.branches
    }


def build_held_out_target_l1_callback(
    n_init: int, n_steps: int, eval_batch_size: int = 8, eval_seed: int = 1000,
    prior_kwargs: dict | None = None, every_n_steps: int | None = None,
) -> Callback:
    """Cheaper, finer-cadence generalization check than the AUC eval above
    (no multi-step rollout under the policy itself, just one held-out
    rollout's oracle targets vs. the current ActionHead's direct
    predictions) -- `l1/exploit`, `l1/explore` (whichever branch(es) the
    trainer has enabled)."""

    def probe(step: int, trainer: Any) -> dict:
        l1 = _held_out_l1(trainer, n_init, n_steps, eval_seed, eval_batch_size, prior_kwargs, blind=False)
        return {f"l1/{branch}": v for branch, v in l1.items()}

    return Callback(name="", fn=probe, every_n_steps=every_n_steps)


def build_blind_ablation_callback(
    n_init: int, n_steps: int, eval_batch_size: int = 8, eval_seed: int = 1001,
    prior_kwargs: dict | None = None, every_n_steps: int | None = None,
) -> Callback:
    """Same held-out L1 check as `build_held_out_target_l1_callback`, with
    `blind=True` -- reports the real/blind ratio per branch
    (`blind_ratio/exploit`, `blind_ratio/explore`), mirroring
    `pipelines/action_head_posterior_distill.py`'s own `real_vs_blind_ratio`
    convention (< 1 means real clearly beats blind, i.e. the cross-attention
    link is carrying signal, not just aux-token/dataset statistics)."""

    def probe(step: int, trainer: Any) -> dict:
        real = _held_out_l1(trainer, n_init, n_steps, eval_seed, eval_batch_size, prior_kwargs, blind=False)
        blind = _held_out_l1(trainer, n_init, n_steps, eval_seed, eval_batch_size, prior_kwargs, blind=True)
        return {
            f"blind_ratio/{branch}": (real[branch] / blind[branch] if blind[branch] else float("nan"))
            for branch in real
        }

    return Callback(name="", fn=probe, every_n_steps=every_n_steps)


def build_explore_signal_rate_callback(
    n_init: int, n_steps: int, eval_batch_size: int = 8, eval_seed: int = 1002,
    prior_kwargs: dict | None = None, build_interesting_points_kwargs: dict | None = None,
    every_n_steps: int | None = None,
) -> Callback:
    """The collapse risk `search/explore.py`'s own docstring flags as real
    and unresolved (once the realized incumbent matches/beats every
    `x_int` point, every improvement weight goes to 0 and `explore_search`
    reports `has_signal=False`) -- `explore/signal_rate`: fraction of
    explore-labeled (instance, step) pairs, in a fresh held-out rollout,
    where `has_signal=True`. Nothing else in the repo currently surfaces
    this number anywhere."""
    from anytimeacquisition.search.explore import explore_search

    build_ip_kwargs = build_interesting_points_kwargs or {"n_sobol": 16, "n_random": 16, "n_basin_restarts": 8}

    def probe(step: int, trainer: Any) -> dict:
        prior = _held_out_prior(trainer.prior.d, eval_batch_size, eval_seed, prior_kwargs)
        rollout = rollout_episode(
            prior, n_init, n_steps, policy_fn=random_policy, build_interesting_points_kwargs=build_ip_kwargs,
        )
        is_exploit = label_branches(rollout["y_context"], n_init)
        is_explore = ~is_exploit

        n_explore_labeled, n_with_signal = 0, 0
        for step_idx, (x_ctx, y_ctx) in enumerate(rollout["pre_step_contexts"]):
            step_mask = is_explore[:, step_idx]
            if not step_mask.any():
                continue
            _, _, has_signal = explore_search(
                prior, trainer.pfn, trainer.bar_dist, x_ctx, y_ctx, rollout["x_int"], rollout["y_int_true"],
                **trainer.explore_search_kwargs,
            )
            n_explore_labeled += int(step_mask.sum().item())
            n_with_signal += int((step_mask & has_signal).sum().item())

        return {"explore/signal_rate": n_with_signal / n_explore_labeled if n_explore_labeled else float("nan")}

    return Callback(name="", fn=probe, every_n_steps=every_n_steps)
