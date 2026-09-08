import pytest
import torch

from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.priors.bo_env import BOEnv

_QT_KWARGS = dict(n_samples=20_000, n_grid=256, n_exact_tail=64, chunk_size=5_000)


def _env(B=4, d=2, budget=8):
    prior = BNNPrior(batch_size=B, x_dim=d, seed=0, cache_dir=None, ecdf_n_draws=5, ecdf_samples_per_draw=100)
    return BOEnv(prior, budget=budget, quantile_table_kwargs=_QT_KWARGS)


def test_reset_shapes_and_n_init():
    env = _env(B=4, d=2)
    D0 = env.reset(n_init=3, seed=0)
    assert D0["x_context"].shape == (4, 3, 2)
    assert D0["y_context"].shape == (4, 3)
    assert env.t == 3


def test_reset_rejects_n_init_over_budget():
    env = _env(budget=5)
    with pytest.raises(AssertionError):
        env.reset(n_init=6)


def test_step_before_reset_raises():
    env = _env()
    with pytest.raises(AssertionError):
        env.step(torch.zeros(4, 2))


def test_step_past_budget_raises():
    env = _env(B=2, d=1, budget=3)
    env.reset(n_init=3, seed=0)
    with pytest.raises(AssertionError):
        env.step(torch.rand(2, 1))


def test_g_reward_is_monotone_nondecreasing_within_episode():
    torch.manual_seed(0)
    B, d, n_init, budget = 6, 2, 3, 12
    env = _env(B=B, d=d, budget=budget)
    env.reset(n_init=n_init, seed=0)

    g_prev = torch.zeros(B)
    while env.t < env.budget:
        x_next = torch.rand(B, d)
        _, g, done = env.step(x_next)
        assert (g >= g_prev - 1e-6).all()
        assert (g >= 0).all() and (g <= 1).all()
        g_prev = g
    assert done


def test_done_flag_exactly_at_budget():
    B, d, n_init, budget = 2, 1, 2, 5
    env = _env(B=B, d=d, budget=budget)
    env.reset(n_init=n_init, seed=0)
    dones = []
    while env.t < env.budget:
        _, _, done = env.step(torch.rand(B, d))
        dones.append(done)
    assert dones[:-1] == [False] * (len(dones) - 1)
    assert dones[-1] is True
    assert env.t == budget


def test_context_grows_by_one_point_per_step_across_batch():
    B, d, n_init = 3, 2, 2
    env = _env(B=B, d=d, budget=n_init + 4)
    env.reset(n_init=n_init, seed=0)
    x_next = torch.rand(B, d)
    D1, _, _ = env.step(x_next)
    assert D1["x_context"].shape == (B, n_init + 1, d)
    assert D1["y_context"].shape == (B, n_init + 1)
    assert torch.allclose(D1["x_context"][:, -1], x_next)


def test_reset_is_reproducible_with_same_seed():
    env_a = _env(B=3, d=2)
    env_b = _env(B=3, d=2)
    D_a = env_a.reset(n_init=4, seed=42)
    D_b = env_b.reset(n_init=4, seed=42)
    assert torch.allclose(D_a["x_context"], D_b["x_context"])
