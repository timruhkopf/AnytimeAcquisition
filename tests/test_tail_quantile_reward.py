import math

import numpy as np
import torch

from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.reward.tail_quantile_reward import (
    _gpd_survival,
    build_ecdf,
    clip_bind_rate,
    clipped_empirical_reward,
    fit_gpd_tail,
    g_from_percentile,
    g_reward_minimize,
    normalized_advantage,
    percentile,
    unclipped_gpd_reward,
)


def test_reward_is_scale_invariant_under_monotone_rescaling():
    """docs/ROADMAP.md §M1 exit criterion: two functions differing by a
    monotone rescaling produce identical reward trajectories for the same
    query sequence -- the reward is purely a function of percentile rank,
    which any monotone transform preserves by construction."""
    rng = np.random.default_rng(0)
    reference = np.sort(rng.normal(size=100_000))
    trajectory = rng.normal(size=20)

    def g_trajectory(ref, traj):
        return np.array([clipped_empirical_reward(f_t, ref, max_score=4.0) for f_t in traj])

    baseline = g_trajectory(reference, trajectory)

    for transform in (lambda y: 3.0 * y + 7.0, lambda y: np.sign(y) * np.abs(y) ** 3, np.exp):
        rescaled_ref = np.sort(transform(reference))
        rescaled_traj = transform(trajectory)
        rescaled = g_trajectory(rescaled_ref, rescaled_traj)
        assert np.allclose(baseline, rescaled, atol=1e-6)


def test_gpd_survival_handles_past_the_finite_endpoint_without_nan():
    """§2.3: for xi < 0, `1 + xi*x/sigma` goes negative past the GPD's
    finite endpoint -- a naive `** (-1/xi)` returns nan/complex there.
    `_gpd_survival` must return exactly 0.0 instead."""
    xi, sigma = -0.3, 1.0
    endpoint = sigma / abs(xi)

    within = _gpd_survival(0.5 * endpoint, xi, sigma)
    assert 0.0 < within <= 1.0
    assert not math.isnan(within)

    past = _gpd_survival(2.0 * endpoint, xi, sigma)
    assert past == 0.0


def test_unclipped_gpd_reward_never_produces_nan_for_adversarial_deep_tail():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=2000)
    u, xi, sigma, p_u = fit_gpd_tail(samples, top_percentile=99.0)

    for f_t in (u + 1e-6, u + 1.0, u + 1e6, u + 1e12):
        score = unclipped_gpd_reward(f_t, samples, u, xi, sigma, p_u)
        assert not math.isnan(score)
        assert not math.isinf(score)


def test_build_ecdf_is_sorted_reproducible_and_per_instance():
    prior = BNNPrior(batch_size=3, x_dim=1, seed=0)
    y_sorted_a = build_ecdf(prior, n_samples=2000, seed=1)
    y_sorted_b = build_ecdf(prior, n_samples=2000, seed=1)

    assert y_sorted_a.shape == (3, 2000)
    assert torch.equal(y_sorted_a, y_sorted_b)  # same seed -> reproducible
    assert (y_sorted_a.diff(dim=1) >= 0).all()  # sorted ascending
    # not identical across batch instances (each is its own architecture draw)
    assert not torch.equal(y_sorted_a[0], y_sorted_a[1])


def test_percentile_matches_hand_computed_rank():
    y_sorted = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
    u = percentile(y_sorted, torch.tensor([2.0]))
    assert torch.allclose(u, torch.tensor([0.5]), atol=1e-6)


def test_normalized_advantage_masks_saturated_states_and_matches_formula():
    g_prev = torch.tensor([0.5, 0.999, 0.9999])
    g_bar = torch.tensor([0.7, 0.9995, 0.99995])
    value, mask = normalized_advantage(g_bar, g_prev, sat_threshold=1 - 1e-3)

    assert mask.tolist() == [True, False, False]
    assert torch.allclose(value[0], torch.tensor((0.7 - 0.5) / (1 - 0.5)), atol=1e-6)


def test_clip_bind_rate_extremes():
    max_score = 4.0
    never_binds = torch.full((100,), 1 - 10 ** (-max_score + 1))  # tail_prob well above the clip threshold
    always_binds = torch.full((100,), 1 - 10 ** (-max_score - 1))  # tail_prob well below it

    assert clip_bind_rate(never_binds, max_score=max_score) == 0.0
    assert clip_bind_rate(always_binds, max_score=max_score) == 1.0


def test_g_from_percentile_matches_clipped_empirical_reward():
    rng = np.random.default_rng(0)
    samples = np.sort(rng.normal(size=50_000))
    f_t = 1.5

    via_samples = clipped_empirical_reward(f_t, samples, max_score=4.0)
    u = np.searchsorted(samples, f_t, side="right") / len(samples)
    via_percentile = g_from_percentile(np.array([u]), max_score=4.0)[0]
    assert abs(via_samples - via_percentile) < 1e-3


def test_g_reward_minimize_rewards_small_values_not_large_ones():
    """The bug this function exists to make impossible to repeat (caught
    2026-09-08 while building M3): composing percentile() + g_from_percentile()
    directly silently computes the reward for *maximizing* v, since
    percentile() is a plain CDF (high for a large v). g_reward_minimize
    must reward SMALL v instead, matching this project's minimize
    convention throughout (priors/bnn.py, search/, gp_acquisition.py, ...)."""
    y_sorted = torch.linspace(0.0, 1.0, 1_000_000).unsqueeze(0)
    small_v = torch.tensor([[1e-4]])  # ~0.01st percentile -> g ~= clip(-log10(1e-4),0,4)/4 = 1.0
    large_v = torch.tensor([[0.99]])  # near the top -- should score poorly

    g_small = g_reward_minimize(y_sorted, small_v)
    g_large = g_reward_minimize(y_sorted, large_v)
    assert g_small.item() > g_large.item()
    assert g_small.item() > 0.9  # near the 0.01st percentile -> near-maximal (4-decade-clipped) reward


def test_g_reward_minimize_matches_manual_flip_of_percentile():
    y_sorted = torch.sort(torch.rand(1, 5000)).values
    v = torch.rand(1, 7)

    expected = g_from_percentile((1.0 - percentile(y_sorted, v)).numpy(), max_score=4.0)
    actual = g_reward_minimize(y_sorted, v).numpy()
    assert np.allclose(actual, expected)
