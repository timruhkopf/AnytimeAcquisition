import torch

from anytimeacquisition.models.bar_distribution import BarDistribution, uniform_bin_borders
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.search.explore import explore_search, improvement_weights


def _tiny_pfn(x_dim=1, seed=0):
    torch.manual_seed(seed)
    pfn = PFN(max_x_dim=x_dim, d_model=16, n_heads=2, n_layers=2, d_ff=32, n_bins=16)
    pfn.eval()
    bar_dist = BarDistribution(uniform_bin_borders(16))
    return pfn, bar_dist


def _tiny_prior(batch_size=3, x_dim=1, seed=0):
    prior = BNNPrior(batch_size=batch_size, x_dim=x_dim, seed=seed)
    prior.reset()
    return prior


def test_improvement_weights_zero_below_incumbent_positive_above():
    incumbent = torch.tensor([0.5, 0.5])
    y_int_true = torch.tensor([[0.1, 0.5, 0.9], [0.4, 0.6, 0.5]])
    w = improvement_weights(incumbent, y_int_true)
    assert w.shape == (2, 3)
    assert (w >= 0.0).all()
    # strictly worse-or-equal-than-incumbent points get exactly zero weight
    assert w[0, 1].item() == 0.0 and w[0, 2].item() == 0.0
    assert w[1, 1].item() == 0.0 and w[1, 2].item() == 0.0
    # strictly better points get positive weight
    assert w[0, 0].item() > 0.0
    assert w[1, 0].item() > 0.0


def test_explore_search_shapes_and_has_signal():
    torch.manual_seed(0)
    x_dim = 1
    pfn, bar_dist = _tiny_pfn(x_dim=x_dim)
    B, Nt, N_int = 3, 5, 6
    prior = _tiny_prior(batch_size=B, x_dim=x_dim)
    x_context, y_context, _, _ = prior.sample_episode(n_train=Nt, n_test=0)
    x_int = torch.rand(B, N_int, x_dim)
    with torch.no_grad():
        y_int_true = prior.evaluate(x_int, noise=False)
    x_realized = torch.rand(B, x_dim)

    x_star, val_star, has_signal = explore_search(
        prior, pfn, bar_dist, x_context, y_context, x_int, y_int_true, x_realized, n_restarts=3, n_steps=5,
    )
    assert x_star.shape == (B, x_dim)
    assert val_star.shape == (B,)
    assert has_signal.shape == (B,)
    assert (x_star >= 0.0).all() and (x_star <= 1.0).all()


def test_explore_search_teacher_forces_x_stars_y_from_the_prior_not_the_pfn():
    """The whole point of the 2026-08-28 fix: x_star's grounding y must come
    from `prior.evaluate`, not from the PFN's own posterior mean at x_star
    under the pre-search context -- assert the two differ (on a randomly
    initialized, untrained PFN they essentially never coincide)."""
    torch.manual_seed(0)
    x_dim = 1
    pfn, bar_dist = _tiny_pfn(x_dim=x_dim)
    prior = _tiny_prior(batch_size=3, x_dim=x_dim)
    x_context, y_context, _, _ = prior.sample_episode(n_train=5, n_test=0)
    x_int = torch.rand(3, 6, x_dim)
    with torch.no_grad():
        y_int_true = prior.evaluate(x_int, noise=False)
    x_realized = torch.rand(3, x_dim)

    x_star, _, has_signal = explore_search(
        prior, pfn, bar_dist, x_context, y_context, x_int, y_int_true, x_realized, n_restarts=4, n_steps=10,
    )
    assert has_signal.any(), "test setup should produce at least one instance with signal"

    with torch.no_grad():
        y_true_at_star = prior.evaluate(x_star.unsqueeze(1), noise=False).squeeze(-1)
        y_pfn_guess_at_star = bar_dist.mean(pfn(x_context, y_context, x_star.unsqueeze(1))).squeeze(-1)
    assert not torch.allclose(y_true_at_star, y_pfn_guess_at_star, atol=1e-3)


def test_explore_search_has_no_signal_when_all_interesting_points_are_worse_than_incumbent():
    torch.manual_seed(0)
    x_dim = 1
    pfn, bar_dist = _tiny_pfn(x_dim=x_dim)
    prior = _tiny_prior(batch_size=2, x_dim=x_dim)
    x_context, y_context = torch.rand(2, 4, x_dim), torch.full((2, 4), 0.01)  # incumbent is already ~best possible
    x_int = torch.rand(2, 5, x_dim)
    y_int_true = torch.full((2, 5), 0.99)  # every interesting point is far worse than the incumbent
    x_realized = torch.rand(2, x_dim)

    _, _, has_signal = explore_search(
        prior, pfn, bar_dist, x_context, y_context, x_int, y_int_true, x_realized, n_restarts=2, n_steps=3,
    )
    assert (~has_signal).all()


def test_explore_search_reduces_weighted_nll_at_interesting_points():
    torch.manual_seed(0)
    x_dim = 1
    pfn, bar_dist = _tiny_pfn(x_dim=x_dim)
    prior = _tiny_prior(batch_size=2, x_dim=x_dim)
    x_context, y_context, _, _ = prior.sample_episode(n_train=5, n_test=0)
    x_int = torch.rand(2, 8, x_dim)
    with torch.no_grad():
        y_int_true = prior.evaluate(x_int, noise=False)
    incumbent = y_context.min(dim=1).values
    weights = improvement_weights(incumbent, y_int_true)

    with torch.no_grad():
        nll_before = bar_dist(pfn(x_context, y_context, x_int), y_int_true)
        weighted_before = (weights * nll_before).sum(dim=-1)
    x_realized = torch.rand(2, x_dim)

    x_star, val_star, has_signal = explore_search(
        prior, pfn, bar_dist, x_context, y_context, x_int, y_int_true, x_realized, n_restarts=6, n_steps=40, lr=0.05,
    )

    with torch.no_grad():
        y_star_true = prior.evaluate(x_star.unsqueeze(1), noise=False)
        x_train_aug = torch.cat([x_context, x_star.unsqueeze(1)], dim=1)
        y_train_aug = torch.cat([y_context, y_star_true], dim=1)
        nll_after = bar_dist(pfn(x_train_aug, y_train_aug, x_int), y_int_true)
        weighted_after = (weights * nll_after).sum(dim=-1)

    for b in range(2):
        if not has_signal[b]:
            continue
        assert weighted_after[b] <= weighted_before[b] + 1e-4
    assert torch.allclose(val_star, weighted_after, atol=1e-4)
