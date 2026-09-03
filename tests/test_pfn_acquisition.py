import torch

from anytimeacquisition.models.bar_distribution import BarDistribution, uniform_bin_borders
from anytimeacquisition.models.baselines.pfn_acquisition import (
    expected_improvement,
    pfn_acquisition_policy,
    pfn_ei_argmax,
)
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.priors.bnn import BNNPrior


def _tiny_pfn(x_dim=1, seed=0):
    torch.manual_seed(seed)
    pfn = PFN(max_x_dim=x_dim, d_model=16, n_heads=2, n_layers=2, d_ff=32, n_bins=16)
    pfn.eval()
    return pfn, pfn.bar_dist


def test_expected_improvement_matches_monte_carlo():
    """Closed-form EI vs. a brute-force Monte Carlo estimate -- the key
    correctness check for the formula adapted (not derived from scratch,
    see module docstring) from PFNs4BO's `archive/src/utils/
    bar_distribution.py::BarDistribution.ei()`. Sample from the
    piecewise-uniform density directly (multinomial bucket choice +
    uniform-within-bucket), not via the PFN -- isolates the formula itself
    from any PFN behavior."""
    torch.manual_seed(0)
    bar_dist = BarDistribution(uniform_bin_borders(n_bins=32))
    logits = torch.randn(5, 32) * 2.0
    incumbent = torch.rand(5)

    closed_form = expected_improvement(bar_dist, logits, incumbent)

    n_samples = 200_000
    p = torch.softmax(logits, -1)
    bucket_idx = torch.multinomial(p, n_samples, replacement=True)  # [5, n_samples]
    lo = bar_dist.borders[:-1][bucket_idx]
    hi = bar_dist.borders[1:][bucket_idx]
    y_samples = lo + (hi - lo) * torch.rand(5, n_samples)
    monte_carlo = (incumbent.unsqueeze(-1) - y_samples).clamp_min(0.0).mean(dim=-1)

    assert torch.allclose(closed_form, monte_carlo, atol=0.01), (closed_form, monte_carlo)


def test_expected_improvement_zero_when_incumbent_below_support():
    """No y in [0,1] can improve on an incumbent already below every bin ->
    EI must be exactly 0 everywhere."""
    bar_dist = BarDistribution(uniform_bin_borders(n_bins=16))
    logits = torch.randn(3, 16)
    incumbent = torch.full((3,), -1.0)  # below borders[0] == 0.0
    ei = expected_improvement(bar_dist, logits, incumbent)
    assert torch.allclose(ei, torch.zeros(3))


def test_expected_improvement_equals_incumbent_minus_mean_when_above_support():
    """When the incumbent is above every bin, every y improves on it by
    exactly (incumbent - y) -- EI collapses to incumbent - E[Y]."""
    bar_dist = BarDistribution(uniform_bin_borders(n_bins=16))
    logits = torch.randn(3, 16)
    incumbent = torch.full((3,), 2.0)  # above borders[-1] == 1.0
    ei = expected_improvement(bar_dist, logits, incumbent)
    expected = incumbent - bar_dist.mean(logits)
    assert torch.allclose(ei, expected, atol=1e-5)


def test_pfn_ei_argmax_shapes_and_self_consistency():
    pfn, bar_dist = _tiny_pfn(x_dim=1)
    prior = BNNPrior(batch_size=4, x_dim=1, seed=1)
    prior.reset()
    x_train, y_train, _, _ = prior.sample_episode(n_train=6, n_test=0)

    x_star, grid, ei_grid = pfn_ei_argmax(pfn, bar_dist, x_train, y_train, n_grid=200)

    assert x_star.shape == (4, 1)
    assert grid.shape == (200, 1)
    assert ei_grid.shape == (4, 200)
    assert (x_star >= 0.0).all() and (x_star <= 1.0).all()

    # x_star must actually be the grid point achieving ei_grid's max, per
    # batch item -- self-consistency between the two returned tensors.
    best_idx = ei_grid.argmax(dim=1)
    assert torch.allclose(x_star.squeeze(-1), grid.squeeze(-1)[best_idx])


def test_pfn_ei_argmax_rejects_multi_dim_x():
    pfn, bar_dist = _tiny_pfn(x_dim=2)
    x_train, y_train = torch.rand(2, 5, 2), torch.rand(2, 5)
    try:
        pfn_ei_argmax(pfn, bar_dist, x_train, y_train)
        assert False, "expected an AssertionError for x_dim != 1"
    except AssertionError:
        pass


def test_pfn_acquisition_policy_matches_policy_fn_contract():
    """Same signature/contract as trainer.exit_rollout.random_policy /
    gp_acquisition_policy -- x_context [B,Nt,x_dim] y_context [B,Nt] ->
    [B,x_dim]."""
    pfn, bar_dist = _tiny_pfn(x_dim=1)
    x_context, y_context = torch.rand(3, 5, 1), torch.rand(3, 5)
    x_next = pfn_acquisition_policy(x_context, y_context, x_dim=1, pfn=pfn, bar_dist=bar_dist, n_grid=100)
    assert x_next.shape == (3, 1)
    assert (x_next >= 0.0).all() and (x_next <= 1.0).all()
