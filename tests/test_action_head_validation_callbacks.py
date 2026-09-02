import math

import pytest
import torch

from anytimeacquisition.callbacks.action_head_validation import (
    build_auc_eval_callback,
    build_blind_ablation_callback,
    build_explore_signal_rate_callback,
    build_held_out_target_l1_callback,
)
from anytimeacquisition.models.action_head import ActionHead, pfn_dims
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.trainer.action_head_imitation_trainer import ActionHeadImitationTrainer


def _fixture_trainer(x_dim=1, branches=("exploit", "explore")):
    torch.manual_seed(0)
    pfn = PFN(max_x_dim=x_dim, d_model=16, n_heads=2, n_layers=1, d_ff=32, n_bins=16)
    pfn.eval()
    pfn_d_model, pfn_n_layers = pfn_dims(pfn)
    action_head = ActionHead(pfn_d_model=pfn_d_model, pfn_n_layers=pfn_n_layers, x_dim=x_dim, d_model=16, n_heads=2, d_ff=32)
    prior = BNNPrior(batch_size=4, x_dim=x_dim, seed=1)
    build_ip_kwargs = {"n_sobol": 4, "n_random": 4, "n_basin_restarts": 2} if "explore" in branches else None
    return ActionHeadImitationTrainer(
        pfn=pfn, bar_dist=pfn.bar_dist, prior=prior, action_head=action_head, branches=list(branches),
        n_init=3, n_steps=5,
        exploit_search_kwargs={"n_restarts": 2, "n_steps": 5},
        explore_search_kwargs={"n_restarts": 2, "n_steps": 5},
        build_interesting_points_kwargs=build_ip_kwargs,
    )


def _all_finite(d: dict) -> bool:
    return all(math.isfinite(v) for v in d.values())


def test_auc_eval_callback_returns_finite_metrics_for_all_three_policies():
    trainer = _fixture_trainer()
    callback = build_auc_eval_callback(
        x_dim=1, n_init=3, n_steps=5, eval_batch_size=3, eval_seed=42,
        ei_kwargs={"num_restarts": 2, "raw_samples": 8}, log_figure=False,
    )
    metrics = callback.fn(0, trainer)
    assert set(metrics) == {
        "auc/action_head", "auc/random", "auc/ei",
        "auc_improvement_vs_random/mean", "auc_improvement_vs_random/std",
        "auc_improvement_vs_random/mean_minus_std", "auc_improvement_vs_random/mean_plus_std",
    }
    assert _all_finite(metrics)
    # rel=1e-5 (not pytest.approx's tighter default): these two values come
    # from different float32 computation orders (tensor subtract-then-mean
    # vs. python-level mean-then-subtract of already-.item()'d floats) --
    # mathematically equivalent, not bit-identical at float32 precision.
    assert metrics["auc_improvement_vs_random/mean"] == pytest.approx(
        metrics["auc/random"] - metrics["auc/action_head"], rel=1e-5
    )


def test_auc_eval_callback_random_baseline_is_memoized_not_recomputed():
    """random/ei are computed once and cached -- a second call on the SAME
    callback must return the identical value without re-rolling anything
    (checked by making random_policy itself non-deterministic across calls
    were it actually re-invoked: reseeding differently before each call
    would change a freshly-computed value but not a cached one)."""
    trainer = _fixture_trainer()
    callback = build_auc_eval_callback(
        x_dim=1, n_init=3, n_steps=5, eval_batch_size=3, eval_seed=42,
        ei_kwargs={"num_restarts": 2, "raw_samples": 8}, log_figure=False,
    )
    torch.manual_seed(0)
    m1 = callback.fn(0, trainer)
    torch.manual_seed(123)  # deliberately different -- would change a fresh recompute
    m2 = callback.fn(0, trainer)
    assert m1["auc/random"] == m2["auc/random"]
    assert m1["auc/ei"] == m2["auc/ei"]


def test_auc_eval_callback_uses_identical_instances_across_policies():
    """Same eval_seed -> same underlying BNN draws for random/action_head/
    ei -- checked across two INDEPENDENTLY memoized callbacks (each caches
    its own baseline once), reseeding the global RNG identically before
    each callback's first call so their own internal random-restart
    generation is comparably seeded too."""
    trainer = _fixture_trainer()

    torch.manual_seed(0)
    callback1 = build_auc_eval_callback(
        x_dim=1, n_init=3, n_steps=5, eval_batch_size=3, eval_seed=42,
        ei_kwargs={"num_restarts": 2, "raw_samples": 8}, log_figure=False,
    )
    m1 = callback1.fn(0, trainer)

    torch.manual_seed(0)
    callback2 = build_auc_eval_callback(
        x_dim=1, n_init=3, n_steps=5, eval_batch_size=3, eval_seed=42,
        ei_kwargs={"num_restarts": 2, "raw_samples": 8}, log_figure=False,
    )
    m2 = callback2.fn(0, trainer)

    assert m1["auc/random"] == m2["auc/random"]


def test_held_out_target_l1_callback_reports_only_enabled_branches():
    trainer = _fixture_trainer(branches=("exploit",))
    callback = build_held_out_target_l1_callback(n_init=3, n_steps=5, eval_batch_size=4, eval_seed=7)
    metrics = callback.fn(0, trainer)
    assert set(metrics) == {"l1/exploit"}
    assert _all_finite(metrics)


def test_blind_ablation_callback_reports_ratio_per_branch():
    # eval_batch_size=16 (not the module default 4) -- build_explore_buffer's
    # require_improvement gate (2026-09-01) can legitimately leave zero
    # surviving explore examples on a small batch/seed combination, which
    # would make blind_ratio/explore nan (division by a zero count) --
    # a bigger batch makes at least one surviving example reliable here.
    trainer = _fixture_trainer(branches=("exploit", "explore"))
    callback = build_blind_ablation_callback(n_init=3, n_steps=5, eval_batch_size=16, eval_seed=8)
    metrics = callback.fn(0, trainer)
    assert set(metrics) == {"blind_ratio/exploit", "blind_ratio/explore"}
    assert _all_finite(metrics)


def test_explore_signal_rate_callback_returns_a_rate_in_zero_one():
    trainer = _fixture_trainer(branches=("explore",))
    callback = build_explore_signal_rate_callback(
        n_init=3, n_steps=8, eval_batch_size=4, eval_seed=9,
        build_interesting_points_kwargs={"n_sobol": 4, "n_random": 4, "n_basin_restarts": 2},
    )
    metrics = callback.fn(0, trainer)
    assert set(metrics) == {"explore/signal_rate"}
    rate = metrics["explore/signal_rate"]
    assert math.isnan(rate) or 0.0 <= rate <= 1.0
