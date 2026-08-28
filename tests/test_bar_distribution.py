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
