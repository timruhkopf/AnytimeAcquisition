import torch

from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.models.surrogates.pfn_surrogate import PFNSurrogate
from anytimeacquisition.oracle.discrete_dp import (
    DiscreteDPOracle,
    build_action_grid,
    discretize_outcomes,
    score_candidate_q,
)
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.reward.tail_quantile_reward import build_ecdf, g_reward_minimize


def _tiny_setup(x_dim=1, seed=0):
    torch.manual_seed(seed)
    pfn = PFN(max_x_dim=x_dim, d_model=16, n_heads=2, n_layers=2, d_ff=32, n_bins=16)
    pfn.eval()
    surrogate = PFNSurrogate(pfn, pfn.bar_dist, use_bf16=False)
    prior = BNNPrior(batch_size=1, x_dim=x_dim, seed=seed)
    y_sorted = build_ecdf(prior, n_samples=5000, seed=1)[0]
    x_context, y_context, _, _ = prior.sample_episode(n_train=2, n_test=0)
    return surrogate, y_sorted, x_context[0], y_context[0]


def test_build_action_grid_shape_and_bounds():
    grid = build_action_grid(d=2, k=4)
    assert grid.shape == (16, 2)
    assert (grid >= 0).all() and (grid <= 1).all()
    # every distinct (i,j) combination of the 4x4 axis grid should appear
    assert len(set(map(tuple, grid.round(decimals=6).tolist()))) == 16


def test_discretize_outcomes_probabilities_sum_to_one_and_bracket_the_support():
    surrogate, *_ = _tiny_setup()
    logits = torch.randn(3, surrogate.bar_dist.num_bars)
    values, probs = discretize_outcomes(surrogate.bar_dist, logits, n_outcome_bins=4)
    assert values.shape == (4,)
    assert probs.shape == (3, 4)
    assert torch.allclose(probs.sum(-1), torch.ones(3), atol=1e-5)
    assert (values >= 0).all() and (values <= 1).all()


def _brute_force_solve(surrogate, y_sorted, grid, n_outcome_bins, max_score, x_context, y_context, budget):
    """Independent, unbatched, unvectorized reimplementation of the same
    recursion `DiscreteDPOracle._solve_batched` computes -- deliberately
    written without any of the batching/broadcasting tricks, so it can
    catch indexing/broadcast bugs in the vectorized version rather than
    share them."""
    incumbent = y_context.min().item()
    n_actions = grid.shape[0]
    with torch.no_grad():
        logits = surrogate.predict(x_context.unsqueeze(0), y_context.unsqueeze(0), grid.unsqueeze(0))[0]  # [n_actions, n_bins]
    values, probs = discretize_outcomes(surrogate.bar_dist, logits, n_outcome_bins)

    Q = torch.zeros(n_actions)
    for a in range(n_actions):
        total = 0.0
        for o in range(n_outcome_bins):
            p_ao = probs[a, o].item()
            y_o = values[o].item()
            next_incumbent = min(incumbent, y_o)
            step_reward = g_reward_minimize(
                y_sorted.unsqueeze(0), torch.tensor([[next_incumbent]]), max_score=max_score
            ).item()
            if budget == 1:
                total += p_ao * step_reward
            else:
                x_child = torch.cat([x_context, grid[a : a + 1]], dim=0)
                y_child = torch.cat([y_context, torch.tensor([y_o])], dim=0)
                _, v_child = _brute_force_solve(
                    surrogate, y_sorted, grid, n_outcome_bins, max_score, x_child, y_child, budget - 1
                )
                total += p_ao * ((step_reward + (budget - 1) * v_child.item()) / budget)
        Q[a] = total
    return Q, Q.max()


def test_solve_matches_independent_brute_force_at_budget_one():
    surrogate, y_sorted, x_context, y_context = _tiny_setup()
    grid = build_action_grid(d=1, k=3)
    oracle = DiscreteDPOracle(surrogate, y_sorted, grid, n_outcome_bins=3)

    q_batched, v_batched = oracle.solve(x_context, y_context, budget=1)
    q_brute, v_brute = _brute_force_solve(surrogate, y_sorted, grid, 3, 4.0, x_context, y_context, budget=1)

    assert torch.allclose(q_batched, q_brute, atol=1e-4)
    assert torch.allclose(v_batched, v_brute, atol=1e-4)


def test_solve_matches_independent_brute_force_at_budget_two():
    surrogate, y_sorted, x_context, y_context = _tiny_setup()
    grid = build_action_grid(d=1, k=3)
    oracle = DiscreteDPOracle(surrogate, y_sorted, grid, n_outcome_bins=3)

    q_batched, v_batched = oracle.solve(x_context, y_context, budget=2)
    q_brute, v_brute = _brute_force_solve(surrogate, y_sorted, grid, 3, 4.0, x_context, y_context, budget=2)

    assert torch.allclose(q_batched, q_brute, atol=1e-4)
    assert torch.allclose(v_batched, v_brute, atol=1e-4)


def test_q_star_is_in_valid_reward_range():
    surrogate, y_sorted, x_context, y_context = _tiny_setup()
    grid = build_action_grid(d=1, k=5)
    oracle = DiscreteDPOracle(surrogate, y_sorted, grid, n_outcome_bins=4)
    q_star, v_star = oracle.solve(x_context, y_context, budget=2)
    assert (q_star >= 0).all() and (q_star <= 1).all()
    assert v_star == q_star.max()


def test_score_candidate_q_against_itself_is_perfect():
    surrogate, y_sorted, x_context, y_context = _tiny_setup()
    grid = build_action_grid(d=1, k=4)
    oracle = DiscreteDPOracle(surrogate, y_sorted, grid, n_outcome_bins=3)
    q_star, _ = oracle.solve(x_context, y_context, budget=2)

    result = score_candidate_q(oracle, q_star, x_context, y_context, budget=2)
    assert result["spearman_rho"] > 0.999
    assert result["regret"] < 1e-5


def test_score_candidate_q_matches_hand_computed_rank_and_regret():
    """score_candidate_q's own math (rank correlation, regret), isolated
    from oracle.solve()'s recursion (already independently verified above)
    -- a hand-specified candidate_q with a known reversed ranking and a
    known suboptimal argmax, checked against manually computed expected
    values."""
    surrogate, y_sorted, x_context, y_context = _tiny_setup()
    grid = build_action_grid(d=1, k=4)
    oracle = DiscreteDPOracle(surrogate, y_sorted, grid, n_outcome_bins=3)
    q_star, v_star = oracle.solve(x_context, y_context, budget=2)

    # exactly reversed ranking of q_star -> perfect NEGATIVE rank correlation
    reversed_rank = (-q_star).argsort().argsort().float()
    result = score_candidate_q(oracle, reversed_rank, x_context, y_context, budget=2)
    assert result["spearman_rho"] < -0.999

    # candidate that deterministically picks the WORST action -> regret
    # should equal exactly V* - Q*[worst action], hand-computed independently
    worst_action = q_star.argmin()
    candidate_favoring_worst = torch.zeros_like(q_star)
    candidate_favoring_worst[worst_action] = 1.0
    result = score_candidate_q(oracle, candidate_favoring_worst, x_context, y_context, budget=2)
    expected_regret = (v_star - q_star[worst_action]).item()
    assert abs(result["regret"] - expected_regret) < 1e-5
    assert result["regret"] > 0  # picking the worst action must never look free
