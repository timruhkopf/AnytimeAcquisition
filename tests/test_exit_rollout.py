import torch

from anytimeacquisition.metrics.inc_auc import incumbent_trajectory
from anytimeacquisition.models.bar_distribution import BarDistribution, uniform_bin_borders
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.trainer.exit_rollout import (
    build_exploit_buffer,
    build_explore_buffer,
    label_branches,
    random_policy,
    rollout_episode,
)


def _tiny_prior(batch_size=4, x_dim=2, seed=0):
    return BNNPrior(batch_size=batch_size, x_dim=x_dim, seed=seed)


def test_rollout_episode_shapes_and_pre_step_context_growth():
    torch.manual_seed(0)
    prior = _tiny_prior(batch_size=3, x_dim=2)
    n_init, n_steps = 4, 6
    rollout = rollout_episode(prior, n_init=n_init, n_steps=n_steps, policy_fn=random_policy)

    assert rollout["x_context"].shape == (3, n_init + n_steps, 2)
    assert rollout["y_context"].shape == (3, n_init + n_steps)
    assert len(rollout["pre_step_contexts"]) == n_steps
    for i, (x_ctx, y_ctx) in enumerate(rollout["pre_step_contexts"]):
        assert x_ctx.shape == (3, n_init + i, 2)
        assert y_ctx.shape == (3, n_init + i)


def test_rollout_episode_resets_exactly_once_per_episode():
    """The whole point of keeping the BNN's state handy for the exploit
    search: `prior.reset()` must fire exactly once per `rollout_episode`
    call (at the start), never again mid-loop -- a second reset mid-episode
    would silently redraw the ground-truth function the trajectory (and any
    oracle search against it) is supposed to be optimizing."""
    torch.manual_seed(0)
    prior = _tiny_prior(batch_size=2, x_dim=2)
    calls = []
    original_reset = prior.reset

    def counting_reset():
        calls.append(1)
        return original_reset()

    prior.reset = counting_reset
    rollout_episode(prior, n_init=3, n_steps=5, policy_fn=random_policy)
    assert len(calls) == 1, "rollout_episode must call prior.reset() exactly once per episode"


def test_label_branches_matches_incumbent_trajectory_reuse():
    n_init = 2
    # Instance 0: improves at step 0, flat at step 1, improves at step 2.
    # Instance 1: never improves after n_init.
    y = torch.tensor([
        [0.5, 0.4, 0.3, 0.3, 0.1],
        [0.5, 0.4, 0.6, 0.7, 0.9],
    ])
    is_exploit = label_branches(y, n_init)
    inc = incumbent_trajectory(y, minimize=True)
    expected = inc[:, n_init:] < inc[:, n_init - 1:-1]
    assert torch.equal(is_exploit, expected)
    assert is_exploit.tolist() == [[True, False, True], [False, False, False]]


def test_build_exploit_buffer_respects_incumbent_and_branch_labels():
    torch.manual_seed(0)
    prior = _tiny_prior(batch_size=3, x_dim=2)
    n_init, n_steps = 4, 8
    rollout = rollout_episode(prior, n_init=n_init, n_steps=n_steps, policy_fn=random_policy)
    is_exploit = label_branches(rollout["y_context"], n_init)

    buffer = build_exploit_buffer(prior, rollout, n_init, exploit_search_kwargs={"n_restarts": 4, "n_steps": 10})

    assert len(buffer) == int(is_exploit.sum().item())
    for ex in buffer:
        assert ex.branch == "exploit"
        assert ex.y_star.item() <= ex.y_context.min().item() + 1e-6
        assert ex.x_star.shape == (2,)


def test_rollout_episode_interesting_points_are_fixed_and_match_prior():
    """build_interesting_points_kwargs must compute x_int/y_int_true right
    after this call's own reset() -- so they must be consistent with the
    SAME instance used for the rest of the episode (checked by
    re-evaluating y_int_true against the live prior after the rollout
    finishes -- if a stray reset() had happened, this would no longer
    match)."""
    torch.manual_seed(0)
    prior = _tiny_prior(batch_size=3, x_dim=2)
    rollout = rollout_episode(
        prior, n_init=3, n_steps=4, policy_fn=random_policy,
        build_interesting_points_kwargs={"n_sobol": 4, "n_random": 4, "n_basin_restarts": 2},
    )
    assert "x_int" in rollout and "y_int_true" in rollout
    assert rollout["x_int"].shape == (3, 10, 2)
    with torch.no_grad():
        expected = prior.evaluate(rollout["x_int"], noise=False)
    assert torch.allclose(rollout["y_int_true"], expected)


def _tiny_pfn(x_dim=2, seed=0):
    torch.manual_seed(seed)
    pfn = PFN(max_x_dim=x_dim, d_model=16, n_heads=2, n_layers=2, d_ff=32, n_bins=16)
    pfn.eval()
    bar_dist = BarDistribution(uniform_bin_borders(16))
    return pfn, bar_dist


def test_build_explore_buffer_covers_exactly_the_explore_labeled_steps_with_signal():
    torch.manual_seed(0)
    x_dim = 2
    prior = _tiny_prior(batch_size=3, x_dim=x_dim)
    pfn, bar_dist = _tiny_pfn(x_dim=x_dim)
    n_init, n_steps = 4, 8
    rollout = rollout_episode(
        prior, n_init=n_init, n_steps=n_steps, policy_fn=random_policy,
        build_interesting_points_kwargs={"n_sobol": 6, "n_random": 6, "n_basin_restarts": 4},
    )
    is_exploit = label_branches(rollout["y_context"], n_init)
    is_explore = ~is_exploit

    buffer = build_explore_buffer(
        prior, pfn, bar_dist, rollout, n_init, explore_search_kwargs={"n_restarts": 3, "n_steps": 5},
    )

    assert len(buffer) <= int(is_explore.sum().item())  # <= because zero-weight (instance, step) pairs are skipped
    for ex in buffer:
        assert ex.branch == "explore"
        assert is_explore[ex.instance_idx, ex.step]
        assert ex.x_star.shape == (x_dim,)
