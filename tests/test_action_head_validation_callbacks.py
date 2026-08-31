import math

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
    assert set(metrics) == {"auc/action_head", "auc/random", "auc/ei"}
    assert _all_finite(metrics)


def test_auc_eval_callback_uses_identical_instances_across_policies():
    """Same eval_seed -> same underlying BNN draws for all three policies
    (see module docstring) -- checked indirectly: two calls with the same
    eval_seed reproduce the exact same auc/random (random_policy's own
    randomness comes from the global torch RNG, so seed it identically
    around each call)."""
    trainer = _fixture_trainer()
    callback = build_auc_eval_callback(
        x_dim=1, n_init=3, n_steps=5, eval_batch_size=3, eval_seed=42,
        ei_kwargs={"num_restarts": 2, "raw_samples": 8}, log_figure=False,
    )
    torch.manual_seed(0)
    m1 = callback.fn(0, trainer)
    torch.manual_seed(0)
    m2 = callback.fn(0, trainer)
    assert m1["auc/random"] == m2["auc/random"]


def test_held_out_target_l1_callback_reports_only_enabled_branches():
    trainer = _fixture_trainer(branches=("exploit",))
    callback = build_held_out_target_l1_callback(n_init=3, n_steps=5, eval_batch_size=4, eval_seed=7)
    metrics = callback.fn(0, trainer)
    assert set(metrics) == {"l1/exploit"}
    assert _all_finite(metrics)


def test_blind_ablation_callback_reports_ratio_per_branch():
    trainer = _fixture_trainer(branches=("exploit", "explore"))
    callback = build_blind_ablation_callback(n_init=3, n_steps=5, eval_batch_size=4, eval_seed=8)
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
