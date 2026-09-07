import torch

from anytimeacquisition.models.baselines.pfn_acquisition import (
    pfn_acquisition_policy,
    pfn_ei_argmax,
)
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.priors.bnn import BNNPrior

# EI's closed form itself now lives on BarDistribution.ei() and is tested in
# tests/test_bar_distribution.py; these tests cover pfn_ei_argmax's grid
# search and pfn_acquisition_policy's rollout_episode contract only.


def _tiny_pfn(x_dim=1, seed=0):
    torch.manual_seed(seed)
    pfn = PFN(max_x_dim=x_dim, d_model=16, n_heads=2, n_layers=2, d_ff=32, n_bins=16)
    pfn.eval()
    return pfn, pfn.bar_dist


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
