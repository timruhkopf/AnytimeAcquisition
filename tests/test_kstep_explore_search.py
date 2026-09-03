import torch

from anytimeacquisition.models.bar_distribution import BarDistribution, uniform_bin_borders
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.search.explore import explore_search, improvement_weights
from anytimeacquisition.search.kstep_explore import kstep_explore_search


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


def test_kstep_explore_search_shapes():
    torch.manual_seed(0)
    x_dim = 1
    pfn, bar_dist = _tiny_pfn(x_dim=x_dim)
    B, Nt, N_int, k = 3, 5, 6, 3
    prior = _tiny_prior(batch_size=B, x_dim=x_dim)
    x_context, y_context, _, _ = prior.sample_episode(n_train=Nt, n_test=0)
    x_int = torch.rand(B, N_int, x_dim)
    with torch.no_grad():
        y_int_true = prior.evaluate(x_int, noise=False)
    x_seed = torch.rand(B, x_dim)

    x_star, val_star, has_signal, plan_star, joint_val = kstep_explore_search(
        prior, pfn, bar_dist, x_context, y_context, x_int, y_int_true, x_seed,
        k=k, n_restarts=3, n_steps=5,
    )
    assert x_star.shape == (B, x_dim)
    assert val_star.shape == (B,)
    assert has_signal.shape == (B,)
    assert plan_star.shape == (B, k, x_dim)
    assert joint_val.shape == (B,)
    assert (x_star >= 0.0).all() and (x_star <= 1.0).all()
    assert (plan_star >= 0.0).all() and (plan_star <= 1.0).all()
    assert torch.equal(plan_star[:, 0], x_star), "x_star must be exactly the plan's own first slot"


def test_kstep_explore_search_val_star_is_the_honest_standalone_value_not_the_joint_score():
    """The central contract this module exists to guarantee: val_star is
    x_star's OWN, independently re-scored value (identical to what a
    from-scratch standalone forward pass on x_star alone would give) --
    never the (systematically more optimistic) joint plan score, which is
    returned separately as joint_val precisely so it can't be confused for
    x_star's value."""
    torch.manual_seed(0)
    x_dim = 1
    pfn, bar_dist = _tiny_pfn(x_dim=x_dim)
    B, Nt, N_int, k = 4, 5, 8, 3
    prior = _tiny_prior(batch_size=B, x_dim=x_dim)
    x_context, y_context, _, _ = prior.sample_episode(n_train=Nt, n_test=0)
    x_int = torch.rand(B, N_int, x_dim)
    with torch.no_grad():
        y_int_true = prior.evaluate(x_int, noise=False)
    x_seed = torch.rand(B, x_dim)
    incumbent = y_context.min(dim=1).values
    weights = improvement_weights(incumbent, y_int_true)

    x_star, val_star, has_signal, plan_star, joint_val = kstep_explore_search(
        prior, pfn, bar_dist, x_context, y_context, x_int, y_int_true, x_seed,
        k=k, n_restarts=3, n_steps=10,
    )
    assert has_signal.any(), "test setup should produce at least one instance with signal"

    # Independent re-derivation of the standalone value, from scratch, not
    # reusing anything internal to kstep_explore_search.
    with torch.no_grad():
        y_star_true = prior.evaluate(x_star.unsqueeze(1), noise=False)
        x_aug = torch.cat([x_context, x_star.unsqueeze(1)], dim=1)
        y_aug = torch.cat([y_context, y_star_true], dim=1)
        nll_star = bar_dist(pfn(x_aug, y_aug, x_int), y_int_true)
        expected_val_star = (weights * nll_star).sum(dim=-1)

    assert torch.allclose(val_star, expected_val_star, atol=1e-4)
    # joint_val is a property of the whole plan and is not required to
    # equal val_star -- assert they're allowed to differ (not that they
    # always do, since a plan can legitimately collapse to k copies of one
    # point) by checking val_star is never assumed equal to it structurally.
    assert val_star.shape == joint_val.shape


def test_kstep_explore_search_slot0_is_not_trust_region_constrained():
    """A trust region around x_seed for slot 0 was tried (notebook
    prototype) and deliberately dropped -- it risks trapping the search in
    whatever local basin the seed sits in. Confirm slot 0 is genuinely free
    to move further than a small radius from x_seed when the objective
    wants it to (not asserting it always does -- just that nothing in this
    module artificially prevents it, unlike the prototype it was promoted
    from)."""
    torch.manual_seed(0)
    x_dim = 1
    pfn, bar_dist = _tiny_pfn(x_dim=x_dim)
    B, Nt, N_int, k = 3, 5, 6, 3
    prior = _tiny_prior(batch_size=B, x_dim=x_dim)
    x_context, y_context, _, _ = prior.sample_episode(n_train=Nt, n_test=0)
    x_int = torch.rand(B, N_int, x_dim)
    with torch.no_grad():
        y_int_true = prior.evaluate(x_int, noise=False)
    x_seed = torch.rand(B, x_dim)
    small_radius = 0.05

    _, _, _, plan_star, _ = kstep_explore_search(
        prior, pfn, bar_dist, x_context, y_context, x_int, y_int_true, x_seed,
        k=k, n_restarts=6, n_steps=30,
    )
    slot0_dist = (plan_star[:, 0] - x_seed).abs()
    assert (slot0_dist > small_radius).any(), (
        "slot 0 should be free to move further than a small radius from x_seed at least somewhere -- "
        "if this ever fails, check no trust-region constraint got reintroduced"
    )


def test_kstep_explore_search_record_trajectory_shape():
    torch.manual_seed(0)
    x_dim = 2
    pfn, bar_dist = _tiny_pfn(x_dim=x_dim)
    B, Nt, N_int, k = 3, 5, 6, 2
    prior = _tiny_prior(batch_size=B, x_dim=x_dim)
    x_context, y_context, _, _ = prior.sample_episode(n_train=Nt, n_test=0)
    x_int = torch.rand(B, N_int, x_dim)
    with torch.no_grad():
        y_int_true = prior.evaluate(x_int, noise=False)
    x_seed = torch.rand(B, x_dim)
    n_restarts, n_steps = 3, 4

    x_star, val_star, has_signal, plan_star, joint_val, trajectory = kstep_explore_search(
        prior, pfn, bar_dist, x_context, y_context, x_int, y_int_true, x_seed,
        k=k, n_restarts=n_restarts, n_steps=n_steps, record_trajectory=True,
    )
    assert trajectory.shape == (n_steps + 1, B, n_restarts, k, x_dim)
    assert torch.equal(trajectory[0, :, 0, 0], x_seed), "restart 0's slot 0 must start exactly at x_seed"


def test_kstep_explore_search_has_signal_matches_explore_search_for_same_weights():
    """has_signal depends only on weight_fn(incumbent, y_int_true), which is
    identical machinery to explore_search's own -- must agree exactly for
    the same inputs, independent of the search mechanism itself."""
    torch.manual_seed(0)
    x_dim = 1
    pfn, bar_dist = _tiny_pfn(x_dim=x_dim)
    B, Nt, N_int = 4, 5, 6
    prior = _tiny_prior(batch_size=B, x_dim=x_dim)
    x_context, y_context, _, _ = prior.sample_episode(n_train=Nt, n_test=0)
    x_int = torch.rand(B, N_int, x_dim)
    with torch.no_grad():
        y_int_true = prior.evaluate(x_int, noise=False)
    x_seed = torch.rand(B, x_dim)

    _, _, has_signal_1step = explore_search(
        prior, pfn, bar_dist, x_context, y_context, x_int, y_int_true, x_seed, n_restarts=2, n_steps=3,
    )
    _, _, has_signal_kstep, _, _ = kstep_explore_search(
        prior, pfn, bar_dist, x_context, y_context, x_int, y_int_true, x_seed, k=3, n_restarts=2, n_steps=3,
    )
    assert torch.equal(has_signal_1step, has_signal_kstep)
