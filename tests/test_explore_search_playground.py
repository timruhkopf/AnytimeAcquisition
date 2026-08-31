import torch

from anytimeacquisition.models.bar_distribution import BarDistribution, uniform_bin_borders
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.pipelines.explore_search_playground import greedy_regret, run_explore_playground


def _tiny_pfn(x_dim=1, seed=0):
    torch.manual_seed(seed)
    pfn = PFN(max_x_dim=x_dim, d_model=16, n_heads=2, n_layers=2, d_ff=32, n_bins=16)
    pfn.eval()
    bar_dist = BarDistribution(uniform_bin_borders(16))
    return pfn, bar_dist


def test_greedy_regret_is_zero_when_argmin_pick_is_the_true_best():
    pfn, bar_dist = _tiny_pfn(x_dim=1)
    x_context, y_context = torch.rand(2, 4, 1), torch.rand(2, 4)
    x_int = torch.rand(2, 5, 1)
    with torch.no_grad():
        predicted_means = bar_dist.mean(pfn(x_context, y_context, x_int))
    greedy_idx = predicted_means.argmin(dim=1)
    # construct y_int_true so the model's own greedy pick IS the true best
    y_int_true = torch.rand(2, 5) + 0.5
    y_int_true[torch.arange(2), greedy_idx] = 0.01

    regret = greedy_regret(pfn, bar_dist, x_context, y_context, x_int, y_int_true)
    assert regret.shape == (2,)
    assert torch.allclose(regret, torch.zeros(2), atol=1e-6)


def test_greedy_regret_is_nonnegative():
    pfn, bar_dist = _tiny_pfn(x_dim=1)
    x_context, y_context = torch.rand(3, 4, 1), torch.rand(3, 4)
    x_int = torch.rand(3, 6, 1)
    y_int_true = torch.rand(3, 6)
    regret = greedy_regret(pfn, bar_dist, x_context, y_context, x_int, y_int_true)
    assert (regret >= -1e-6).all()


def test_run_explore_playground_smoke_and_metric_shapes():
    torch.manual_seed(0)
    x_dim = 1
    pfn, bar_dist = _tiny_pfn(x_dim=x_dim)

    result = run_explore_playground(
        pfn=pfn, bar_dist=bar_dist, x_dim=x_dim, prior_batch_size=2, seed=0,
        n_episodes=2, n_init=3, n_steps=4,
        n_sobol=2, n_random=2, n_basin_restarts=2,
        explore_n_restarts=2, explore_n_steps=3, explore_lr=0.05,
    )

    expected_keys = {
        "n_examples", "mean_regret_before", "mean_regret_after", "mean_regret_reduction",
        "frac_regret_improved", "frac_regret_worsened", "mean_weighted_nll_improvement",
        "nll_improvement_vs_regret_reduction_corr", "regret_before", "regret_after",
        "weighted_nll_before", "weighted_nll_after",
    }
    assert expected_keys.issubset(result.keys())
    n = result["n_examples"]
    assert len(result["regret_before"]) == n
    assert len(result["regret_after"]) == n
    assert all(r >= -1e-6 for r in result["regret_before"])
    assert all(r >= -1e-6 for r in result["regret_after"])
    assert 0.0 <= result["frac_regret_improved"] <= 1.0
    assert 0.0 <= result["frac_regret_worsened"] <= 1.0
