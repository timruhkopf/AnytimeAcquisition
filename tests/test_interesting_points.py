import torch

from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.search.interesting_points import build_interesting_points, find_basins


def _tiny_prior(batch_size=3, x_dim=2, seed=0):
    prior = BNNPrior(batch_size=batch_size, x_dim=x_dim, seed=seed)
    prior.reset()
    return prior


def test_find_basins_shape_and_range():
    torch.manual_seed(0)
    prior = _tiny_prior(batch_size=4, x_dim=2)
    x = find_basins(prior, n_restarts=6, n_steps=5)
    assert x.shape == (4, 6, 2)
    assert (x >= 0.0).all() and (x <= 1.0).all()
    assert not x.requires_grad, "returned basin points must be detached"


def test_build_interesting_points_shapes_and_matches_prior_evaluate():
    torch.manual_seed(0)
    prior = _tiny_prior(batch_size=3, x_dim=2)
    n_sobol, n_random, n_basin = 5, 5, 4
    x_int, y_int_true = build_interesting_points(
        prior, n_sobol=n_sobol, n_random=n_random, n_basin_restarts=n_basin,
    )
    n_total = n_sobol + n_random + n_basin
    assert x_int.shape == (3, n_total, 2)
    assert y_int_true.shape == (3, n_total)
    with torch.no_grad():
        expected = prior.evaluate(x_int, noise=False)
    assert torch.allclose(y_int_true, expected)


def test_build_interesting_points_sobol_component_is_shared_across_instances():
    """The Sobol draw itself doesn't depend on which instance it covers --
    only the random and basin components should differ per instance."""
    torch.manual_seed(0)
    prior = _tiny_prior(batch_size=3, x_dim=2)
    x_int, _ = build_interesting_points(prior, n_sobol=8, n_random=0, n_basin_restarts=0, sobol_seed=1)
    assert torch.allclose(x_int[0], x_int[1])
    assert torch.allclose(x_int[0], x_int[2])


def test_basin_restarts_land_close_to_a_dense_grid_optimum():
    torch.manual_seed(0)
    prior = _tiny_prior(batch_size=2, x_dim=1)
    x_int, y_int_true = build_interesting_points(prior, n_sobol=0, n_random=0, n_basin_restarts=10)

    grid = torch.linspace(0.0, 1.0, 500).view(1, -1, 1).expand(prior.B, -1, -1)
    with torch.no_grad():
        grid_best = prior.evaluate(grid, noise=False).min(dim=1).values
    basin_best = y_int_true.min(dim=1).values
    assert (basin_best - grid_best).mean().item() < 0.02
