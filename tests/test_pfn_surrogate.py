import torch

from anytimeacquisition.metrics.rollout import rollout_episode
from anytimeacquisition.models.baselines.pfn_acquisition import expected_improvement
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.models.surrogates.pfn_surrogate import (
    PFNSurrogate,
    pfn_surrogate_ei_policy,
    probability_of_improvement,
)
from anytimeacquisition.priors.bnn import BNNPrior


def _tiny_pfn(x_dim=2, seed=0):
    torch.manual_seed(seed)
    pfn = PFN(max_x_dim=x_dim, d_model=16, n_heads=2, n_layers=2, d_ff=32, n_bins=16)
    pfn.eval()
    return pfn, pfn.bar_dist


def test_predict_shape_and_wrapper_matches_calling_pfn_directly():
    pfn, bar_dist = _tiny_pfn(x_dim=2)
    surrogate = PFNSurrogate(pfn, bar_dist, use_bf16=False)
    x_context, y_context = torch.rand(3, 4, 2), torch.rand(3, 4)
    candidates = torch.rand(3, 5, 2)

    logits = surrogate.predict(x_context, y_context, candidates)
    assert logits.shape == (3, 5, 16)

    with torch.no_grad():
        n_features = x_context.new_full((3,), 2)
        expected_logits = pfn(x_context, y_context, candidates, n_features=n_features)
    assert torch.allclose(logits, expected_logits, atol=1e-5)


def test_expected_improvement_matches_the_shared_pfn_acquisition_function():
    pfn, bar_dist = _tiny_pfn(x_dim=1)
    surrogate = PFNSurrogate(pfn, bar_dist, use_bf16=False)
    x_context, y_context = torch.rand(2, 3, 1), torch.rand(2, 3)
    candidates = torch.rand(2, 7, 1)

    ei, logits = surrogate.expected_improvement(x_context, y_context, candidates)
    ei_direct = expected_improvement(bar_dist, logits, y_context.min(dim=1).values.unsqueeze(-1))
    assert torch.allclose(ei, ei_direct)
    assert (ei >= 0).all()  # EI is non-negative by construction


def test_probability_of_improvement_is_a_valid_cdf():
    pfn, bar_dist = _tiny_pfn(x_dim=1)
    surrogate = PFNSurrogate(pfn, bar_dist, use_bf16=False)
    x_context, y_context = torch.rand(2, 3, 1), torch.rand(2, 3)

    pi_at_zero, _ = surrogate.probability_of_improvement(
        x_context, y_context, torch.rand(2, 1, 1), threshold=torch.zeros(2)
    )
    pi_at_one, _ = surrogate.probability_of_improvement(
        x_context, y_context, torch.rand(2, 1, 1), threshold=torch.ones(2)
    )
    assert torch.allclose(pi_at_zero, torch.zeros(2, 1), atol=1e-5)  # nothing below the support's lower edge
    assert torch.allclose(pi_at_one, torch.ones(2, 1), atol=1e-5)  # everything below the upper edge


def test_probability_of_improvement_matches_monte_carlo():
    torch.manual_seed(0)
    from anytimeacquisition.models.bar_distribution import BarDistribution, uniform_bin_borders

    bar_dist = BarDistribution(uniform_bin_borders(n_bins=32))
    logits = torch.randn(5, 32) * 2.0
    threshold = torch.rand(5)

    closed_form = probability_of_improvement(bar_dist, logits, threshold)

    n_samples = 200_000
    p = torch.softmax(logits, -1)
    bucket_idx = torch.multinomial(p, n_samples, replacement=True)
    lo, hi = bar_dist.borders[:-1][bucket_idx], bar_dist.borders[1:][bucket_idx]
    y_samples = lo + (hi - lo) * torch.rand(5, n_samples)
    mc = (y_samples < threshold.unsqueeze(-1)).float().mean(dim=1)

    assert torch.allclose(closed_form, mc, atol=0.01)


def test_pfn_surrogate_ei_policy_is_a_drop_in_rollout_policy_fn():
    from functools import partial

    pfn, bar_dist = _tiny_pfn(x_dim=2)
    surrogate = PFNSurrogate(pfn, bar_dist, use_bf16=False)
    prior = BNNPrior(batch_size=3, x_dim=2, seed=0)

    policy_fn = partial(pfn_surrogate_ei_policy, surrogate=surrogate, n_candidates=32, seed=1)
    rollout = rollout_episode(prior, n_init=3, n_steps=4, policy_fn=policy_fn)
    assert rollout["x_context"].shape == (3, 7, 2)
    assert (rollout["x_context"] >= 0).all() and (rollout["x_context"] <= 1).all()
