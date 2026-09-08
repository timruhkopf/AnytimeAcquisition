import math

import torch

from anytimeacquisition.models.bar_distribution import BarDistribution, uniform_bin_borders


def test_nll_under_uniform_logits_is_zero_density_one():
    bd = BarDistribution(uniform_bin_borders(n_bins=64))
    logits = torch.zeros(10, 64)
    y = torch.rand(10)
    nll = bd(logits, y)
    assert torch.allclose(nll, torch.zeros_like(nll), atol=1e-5)


def test_entropy_of_uniform_matches_analytic_uniform_distribution():
    # Differential entropy of U(0,1) is log(1-0) = 0.
    bd = BarDistribution(uniform_bin_borders(n_bins=64))
    logits = torch.zeros(5, 64)
    h = bd.entropy(logits)
    assert torch.allclose(h, torch.zeros_like(h), atol=1e-5)


def test_entropy_is_lower_for_a_confident_distribution():
    bd = BarDistribution(uniform_bin_borders(n_bins=64))
    uniform_logits = torch.zeros(1, 64)
    confident_logits = torch.full((1, 64), -10.0)
    confident_logits[0, 5] = 20.0
    assert bd.entropy(confident_logits).item() < bd.entropy(uniform_logits).item()


def test_mean_under_uniform_logits_is_midpoint():
    bd = BarDistribution(uniform_bin_borders(n_bins=64))
    logits = torch.zeros(3, 64)
    assert torch.allclose(bd.mean(logits), torch.full((3,), 0.5), atol=1e-3)


def test_nll_is_differentiable_wrt_logits():
    bd = BarDistribution(uniform_bin_borders(n_bins=32))
    logits = torch.randn(4, 32, requires_grad=True)
    y = torch.rand(4)
    loss = bd(logits, y).mean()
    loss.backward()
    assert logits.grad is not None
    assert (logits.grad.abs() > 0).any()


def test_ei_matches_monte_carlo():
    """Closed-form EI vs. a brute-force Monte Carlo estimate -- the key
    correctness check for the formula ported (not derived from scratch,
    see module docstring) from PFNs4BO's `archive/src/utils/
    bar_distribution.py::BarDistribution.ei()`. Sample from the
    piecewise-uniform density directly (multinomial bucket choice +
    uniform-within-bucket), not via a PFN -- isolates the formula itself."""
    torch.manual_seed(0)
    bd = BarDistribution(uniform_bin_borders(n_bins=32))
    logits = torch.randn(5, 32) * 2.0
    best_f = torch.rand(5)

    closed_form = bd.ei(logits, best_f)

    n_samples = 200_000
    p = torch.softmax(logits, -1)
    bucket_idx = torch.multinomial(p, n_samples, replacement=True)  # [5, n_samples]
    lo, hi = bd.borders[:-1][bucket_idx], bd.borders[1:][bucket_idx]
    y_samples = lo + (hi - lo) * torch.rand(5, n_samples)
    monte_carlo = (best_f.unsqueeze(-1) - y_samples).clamp_min(0.0).mean(dim=-1)

    assert torch.allclose(closed_form, monte_carlo, atol=0.01), (closed_form, monte_carlo)


def test_ei_zero_when_best_f_below_support():
    """No y in [0,1] can improve on a best_f already below every bin -> EI
    must be exactly 0 everywhere."""
    bd = BarDistribution(uniform_bin_borders(n_bins=16))
    logits = torch.randn(3, 16)
    best_f = torch.full((3,), -1.0)  # below borders[0] == 0.0
    assert torch.allclose(bd.ei(logits, best_f), torch.zeros(3))


def test_ei_equals_best_f_minus_mean_when_above_support():
    """When best_f is above every bin, every y improves on it by exactly
    (best_f - y) -- EI collapses to best_f - E[Y]."""
    bd = BarDistribution(uniform_bin_borders(n_bins=16))
    logits = torch.randn(3, 16)
    best_f = torch.full((3,), 2.0)  # above borders[-1] == 1.0
    assert torch.allclose(bd.ei(logits, best_f), best_f - bd.mean(logits), atol=1e-5)


def test_ei_rejects_best_f_missing_the_broadcast_dim():
    """The exact bug found 2026-09-08 (notebooks/vla_readout_ei_probe.ipynb):
    passing best_f=[B] against 3D logits=[B,Q,n_bins] without pre-unsqueezing
    silently mis-broadcasts (using a DIFFERENT batch item's threshold) rather
    than raising, whenever B happens to equal Q. Must now raise instead."""
    bd = BarDistribution(uniform_bin_borders(n_bins=8))
    logits = torch.randn(4, 4, 8)  # B == Q == 4, the exact coincidence that hid the bug
    best_f_missing_dim = torch.rand(4)
    try:
        bd.ei(logits, best_f_missing_dim)
        assert False, "expected an assertion error for best_f missing the broadcast dim"
    except AssertionError as e:
        assert "unsqueeze" in str(e)

    # Correct usage: pre-unsqueeze so best_f is shared across the query axis.
    result = bd.ei(logits, best_f_missing_dim.unsqueeze(-1))
    assert result.shape == (4, 4)


def test_ei_with_per_env_threshold_matches_per_point_call():
    """One threshold per env, shared across Q query points (the
    LayerLockedReadout notebook's actual use case) must give the same
    result as calling ei() once per query point with that env's own
    threshold -- catches any cross-env mixups in the broadcast."""
    torch.manual_seed(0)
    bd = BarDistribution(uniform_bin_borders(n_bins=16))
    B, Q = 3, 5
    logits = torch.randn(B, Q, 16)
    best_f = torch.rand(B)

    batched = bd.ei(logits, best_f.unsqueeze(-1))  # [B, Q]
    for b in range(B):
        per_point = bd.ei(logits[b], best_f[b].expand(Q))  # [Q], using env b's own threshold
        assert torch.allclose(batched[b], per_point, atol=1e-6)


def test_pi_matches_monte_carlo():
    torch.manual_seed(0)
    bd = BarDistribution(uniform_bin_borders(n_bins=32))
    logits = torch.randn(5, 32) * 2.0
    best_f = torch.rand(5)

    closed_form = bd.pi(logits, best_f)

    n_samples = 200_000
    p = torch.softmax(logits, -1)
    bucket_idx = torch.multinomial(p, n_samples, replacement=True)
    lo, hi = bd.borders[:-1][bucket_idx], bd.borders[1:][bucket_idx]
    y_samples = lo + (hi - lo) * torch.rand(5, n_samples)
    monte_carlo = (y_samples < best_f.unsqueeze(-1)).float().mean(dim=-1)

    assert torch.allclose(closed_form, monte_carlo, atol=0.01)


def test_pi_is_a_valid_cdf_at_the_supports_edges():
    bd = BarDistribution(uniform_bin_borders(n_bins=16))
    logits = torch.randn(4, 16)
    assert torch.allclose(bd.pi(logits, torch.zeros(4)), torch.zeros(4), atol=1e-5)
    assert torch.allclose(bd.pi(logits, torch.ones(4)), torch.ones(4), atol=1e-5)


def test_quantile_matches_icdf_and_brackets_the_median():
    bd = BarDistribution(uniform_bin_borders(n_bins=64))
    logits = torch.randn(3, 64)
    lo, hi = bd.quantile(logits, center_prob=0.682).unbind(-1)
    assert torch.allclose(lo, bd.icdf(logits, (1 - 0.682) / 2))
    assert torch.allclose(hi, bd.icdf(logits, 1 - (1 - 0.682) / 2))
    median = bd.median(logits)
    assert (lo <= median).all() and (median <= hi).all()


def test_ucb_matches_lower_quantile():
    bd = BarDistribution(uniform_bin_borders(n_bins=64))
    logits = torch.randn(3, 64)
    rest_prob = 0.1
    assert torch.allclose(bd.ucb(logits, rest_prob=rest_prob), bd.icdf(logits, rest_prob))
