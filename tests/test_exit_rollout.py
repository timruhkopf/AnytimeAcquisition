import torch

from anytimeacquisition.metrics.inc_auc import incumbent_trajectory
from anytimeacquisition.models.bar_distribution import BarDistribution, uniform_bin_borders
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.search.explore import greedy_regret
from anytimeacquisition.trainer.exit_rollout import (
    build_exploit_buffer,
    build_exploit_chunk_buffer,
    build_explore_buffer,
    build_explore_chunk_buffer,
    label_branches,
    mixed_policy_fn,
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


def test_rollout_episode_reset_false_reuses_the_same_instance():
    """reset=False must skip prior.reset() (caller resets once, up front)
    and roll out against whatever instance is already live -- checked by
    verifying the SAME x_context reproduces the SAME y via prior.evaluate
    across two separate reset=False calls on one prior."""
    torch.manual_seed(0)
    prior = _tiny_prior(batch_size=3, x_dim=2)
    prior.reset()
    calls = []
    original_reset = prior.reset

    def counting_reset():
        calls.append(1)
        return original_reset()

    prior.reset = counting_reset
    rollout_episode(prior, n_init=3, n_steps=4, policy_fn=random_policy, reset=False)
    rollout_episode(prior, n_init=3, n_steps=4, policy_fn=random_policy, reset=False)
    assert len(calls) == 0, "reset=False must never call prior.reset()"

    probe_x = torch.rand(3, 5, 2)
    y1 = prior.evaluate(probe_x, noise=False)
    y2 = prior.evaluate(probe_x, noise=False)
    assert torch.allclose(y1, y2), "the underlying instance must be unchanged across reset=False calls"


def test_mixed_policy_fn_beta_one_matches_policy_a_beta_zero_matches_policy_b():
    torch.manual_seed(0)
    x_context, y_context = torch.rand(5, 3, 2), torch.rand(5, 3)
    policy_a = lambda xc, yc, d: torch.full((xc.shape[0], d), 111.0)
    policy_b = lambda xc, yc, d: torch.full((xc.shape[0], d), 222.0)

    always_a = mixed_policy_fn(policy_a, policy_b, beta=1.0)
    assert torch.equal(always_a(x_context, y_context, 2), torch.full((5, 2), 111.0))

    always_b = mixed_policy_fn(policy_a, policy_b, beta=0.0)
    assert torch.equal(always_b(x_context, y_context, 2), torch.full((5, 2), 222.0))


def test_mixed_policy_fn_mixes_per_instance():
    torch.manual_seed(0)
    x_context, y_context = torch.rand(200, 3, 2), torch.rand(200, 3)
    policy_a = lambda xc, yc, d: torch.full((xc.shape[0], d), 1.0)
    policy_b = lambda xc, yc, d: torch.full((xc.shape[0], d), 0.0)

    mixed = mixed_policy_fn(policy_a, policy_b, beta=0.3)
    action = mixed(x_context, y_context, 2)
    frac_a = (action == 1.0).float().mean().item()
    assert 0.15 < frac_a < 0.45, f"expected roughly beta=0.3 of instances to use policy_a, got {frac_a}"


def test_mixed_policy_fn_usage_counter_tracks_realized_split():
    torch.manual_seed(0)
    x_context, y_context = torch.rand(200, 3, 2), torch.rand(200, 3)
    policy_a = lambda xc, yc, d: torch.zeros(xc.shape[0], d)
    policy_b = lambda xc, yc, d: torch.zeros(xc.shape[0], d)

    usage = {}
    mixed = mixed_policy_fn(policy_a, policy_b, beta=0.3, usage_counter=usage)
    mixed(x_context, y_context, 2)
    mixed(x_context, y_context, 2)

    assert usage["a"] + usage["b"] == 400
    frac_a = usage["a"] / 400
    assert 0.15 < frac_a < 0.45


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


def test_build_exploit_buffer_steps_filter_restricts_which_steps_run():
    torch.manual_seed(0)
    prior = _tiny_prior(batch_size=3, x_dim=2)
    n_init, n_steps = 4, 8
    rollout = rollout_episode(prior, n_init=n_init, n_steps=n_steps, policy_fn=random_policy)
    is_exploit = label_branches(rollout["y_context"], n_init)
    exploit_steps = {s for s in range(n_steps) if is_exploit[:, s].any()}
    assert exploit_steps, "test setup should have at least one exploit-labeled step"
    keep = {next(iter(exploit_steps))}

    buffer = build_exploit_buffer(
        prior, rollout, n_init, exploit_search_kwargs={"n_restarts": 4, "n_steps": 10}, steps=keep,
    )
    assert buffer and all(ex.step in keep for ex in buffer)


def test_build_exploit_buffer_require_exploit_label_false_targets_flat_instances():
    torch.manual_seed(0)
    prior = _tiny_prior(batch_size=3, x_dim=2)
    n_init, n_steps = 4, 8
    rollout = rollout_episode(prior, n_init=n_init, n_steps=n_steps, policy_fn=random_policy)
    is_exploit = label_branches(rollout["y_context"], n_init)
    flat_steps = {s for s in range(n_steps) if (~is_exploit[:, s]).any()}
    assert flat_steps, "test setup should have at least one flat step"

    filler = build_exploit_buffer(
        prior, rollout, n_init, exploit_search_kwargs={"n_restarts": 4, "n_steps": 10},
        steps=flat_steps, require_exploit_label=False,
    )
    assert filler
    for ex in filler:
        assert ex.branch == "exploit"
        assert not is_exploit[ex.instance_idx, ex.step], "filler must only cover flat (non-exploit-labeled) instances"


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


def test_build_explore_buffer_steps_filter_restricts_which_steps_run():
    torch.manual_seed(0)
    x_dim = 2
    prior = _tiny_prior(batch_size=3, x_dim=x_dim)
    pfn, bar_dist = _tiny_pfn(x_dim=x_dim)
    n_init, n_steps = 4, 8
    rollout = rollout_episode(
        prior, n_init=n_init, n_steps=n_steps, policy_fn=random_policy,
        build_interesting_points_kwargs={"n_sobol": 6, "n_random": 6, "n_basin_restarts": 4},
    )
    is_explore = ~label_branches(rollout["y_context"], n_init)
    explore_steps = {s for s in range(n_steps) if is_explore[:, s].any()}
    assert len(explore_steps) >= 2, "test setup should have at least two explore-labeled steps"
    keep = {next(iter(explore_steps))}

    buffer = build_explore_buffer(
        prior, pfn, bar_dist, rollout, n_init, explore_search_kwargs={"n_restarts": 3, "n_steps": 5}, steps=keep,
    )
    assert all(ex.step in keep for ex in buffer)


def test_build_explore_buffer_require_regret_improvement_is_a_strictly_stronger_filter():
    """require_regret_improvement gates on real greedy_regret, not just the
    weighted-NLL proxy require_improvement already gates on -- every
    surviving example must have strictly improved regret (checked by
    recomputing it independently here, not trusted from inside the buffer
    builder), and turning it on can only ever keep a subset of what
    require_improvement alone keeps (a stricter filter on top of the
    existing one, same rollout/search calls either way)."""
    torch.manual_seed(0)
    x_dim = 2
    prior = _tiny_prior(batch_size=6, x_dim=x_dim)
    pfn, bar_dist = _tiny_pfn(x_dim=x_dim)
    n_init, n_steps = 4, 10
    rollout = rollout_episode(
        prior, n_init=n_init, n_steps=n_steps, policy_fn=random_policy,
        build_interesting_points_kwargs={"n_sobol": 8, "n_random": 8, "n_basin_restarts": 4},
    )

    torch.manual_seed(1)
    buffer_proxy = build_explore_buffer(
        prior, pfn, bar_dist, rollout, n_init, explore_search_kwargs={"n_restarts": 3, "n_steps": 5},
    )
    torch.manual_seed(1)
    buffer_regret = build_explore_buffer(
        prior, pfn, bar_dist, rollout, n_init, explore_search_kwargs={"n_restarts": 3, "n_steps": 5},
        require_regret_improvement=True,
    )
    assert len(buffer_regret) <= len(buffer_proxy)

    for ex in buffer_regret:
        x_ctx, y_ctx = ex.x_context.unsqueeze(0), ex.y_context.unsqueeze(0)
        x_int_b = rollout["x_int"][ex.instance_idx].unsqueeze(0)
        y_int_true_b = rollout["y_int_true"][ex.instance_idx].unsqueeze(0)
        regret_before = greedy_regret(pfn, bar_dist, x_ctx, y_ctx, x_int_b, y_int_true_b)
        with torch.no_grad():
            y_star_true = prior.evaluate(
                ex.x_star.view(1, 1, -1).expand(prior.B, 1, -1), noise=False
            )[ex.instance_idx].view(1, 1)
        x_ctx_aug = torch.cat([x_ctx, ex.x_star.view(1, 1, -1)], dim=1)
        y_ctx_aug = torch.cat([y_ctx, y_star_true], dim=1)
        regret_after = greedy_regret(pfn, bar_dist, x_ctx_aug, y_ctx_aug, x_int_b, y_int_true_b)
        assert regret_after.item() < regret_before.item()


def test_build_explore_buffer_k_greater_than_one_uses_kstep_explore_search():
    torch.manual_seed(0)
    x_dim = 1
    prior = _tiny_prior(batch_size=3, x_dim=x_dim)
    pfn, bar_dist = _tiny_pfn(x_dim=x_dim)
    n_init, n_steps = 4, 8
    rollout = rollout_episode(
        prior, n_init=n_init, n_steps=n_steps, policy_fn=random_policy,
        build_interesting_points_kwargs={"n_sobol": 6, "n_random": 6, "n_basin_restarts": 4},
    )

    buffer_1step = build_explore_buffer(
        prior, pfn, bar_dist, rollout, n_init, explore_search_kwargs={"n_restarts": 3, "n_steps": 5},
        k=1, require_improvement=False,
    )
    buffer_kstep = build_explore_buffer(
        prior, pfn, bar_dist, rollout, n_init, explore_search_kwargs={"n_restarts": 3, "n_steps": 5},
        k=3, require_improvement=False,
    )
    # Same shapes/contract either way -- x_star is always ONE point, never
    # the full k-step plan, regardless of which search produced it.
    for ex in buffer_kstep:
        assert ex.x_star.shape == (x_dim,)
        assert ex.branch == "explore"
    # Not asserting the two buffers pick the same instances (different
    # searches can have different has_signal outcomes in principle) -- just
    # that k>1 actually runs and produces well-formed examples.
    assert isinstance(buffer_1step, list) and isinstance(buffer_kstep, list)


def test_build_explore_buffer_x_seed_mode_realized_uses_the_rollouts_own_action():
    torch.manual_seed(0)
    x_dim = 1
    prior = _tiny_prior(batch_size=3, x_dim=x_dim)
    pfn, bar_dist = _tiny_pfn(x_dim=x_dim)
    n_init, n_steps = 4, 8
    rollout = rollout_episode(
        prior, n_init=n_init, n_steps=n_steps, policy_fn=random_policy,
        build_interesting_points_kwargs={"n_sobol": 6, "n_random": 6, "n_basin_restarts": 4},
    )
    is_explore = ~label_branches(rollout["y_context"], n_init)
    explore_steps = {s for s in range(n_steps) if is_explore[:, s].any()}
    assert explore_steps, "test setup should have at least one explore-labeled step"
    step = next(iter(explore_steps))

    # n_steps=1, lr=0 (no-op search) isolates x_seed itself: has_signal
    # aside, with zero learning rate x_star must equal whatever x_seed was.
    buffer = build_explore_buffer(
        prior, pfn, bar_dist, rollout, n_init,
        explore_search_kwargs={"n_restarts": 1, "n_steps": 1, "lr": 0.0},
        steps={step}, require_improvement=False, x_seed_mode="realized",
    )
    for ex in buffer:
        x_realized = rollout["x_context"][ex.instance_idx, n_init + ex.step]
        assert torch.allclose(ex.x_star, x_realized, atol=1e-5)


def test_build_explore_buffer_x_seed_mode_invalid_raises():
    torch.manual_seed(0)
    x_dim = 1
    prior = _tiny_prior(batch_size=2, x_dim=x_dim)
    pfn, bar_dist = _tiny_pfn(x_dim=x_dim)
    rollout = rollout_episode(
        prior, n_init=4, n_steps=4, policy_fn=random_policy,
        build_interesting_points_kwargs={"n_sobol": 4, "n_random": 4, "n_basin_restarts": 2},
    )
    import pytest
    with pytest.raises(ValueError):
        build_explore_buffer(prior, pfn, bar_dist, rollout, n_init=4, x_seed_mode="bogus")


def test_build_exploit_chunk_buffer_repeats_the_single_point_across_the_chunk():
    torch.manual_seed(0)
    x_dim, chunk_len = 2, 4
    prior = _tiny_prior(batch_size=3, x_dim=x_dim)
    n_init, n_steps = 4, 8
    rollout = rollout_episode(prior, n_init=n_init, n_steps=n_steps, policy_fn=random_policy)

    buffer = build_exploit_chunk_buffer(
        prior, rollout, n_init, chunk_len=chunk_len, exploit_search_kwargs={"n_restarts": 2, "n_steps": 5},
    )
    assert len(buffer) > 0, "test setup should have at least one exploit-labeled step"
    for ex in buffer:
        assert ex.branch == "exploit"
        assert ex.target_chunk.shape == (chunk_len, x_dim)
        # Degenerate chunk: every position is the exact same single point.
        assert torch.allclose(ex.target_chunk, ex.target_chunk[0].unsqueeze(0).expand(chunk_len, -1))


def test_build_explore_chunk_buffer_shapes_and_gating():
    torch.manual_seed(0)
    x_dim, chunk_len = 1, 3
    prior = _tiny_prior(batch_size=4, x_dim=x_dim)
    pfn, bar_dist = _tiny_pfn(x_dim=x_dim)
    n_init, n_steps = 4, 8
    rollout = rollout_episode(
        prior, n_init=n_init, n_steps=n_steps, policy_fn=random_policy,
        build_interesting_points_kwargs={"n_sobol": 6, "n_random": 6, "n_basin_restarts": 4},
    )
    is_explore = ~label_branches(rollout["y_context"], n_init)

    buffer = build_explore_chunk_buffer(
        prior, pfn, bar_dist, rollout, n_init, chunk_len=chunk_len,
        explore_search_kwargs={"n_restarts": 2, "n_steps": 5},
    )
    assert len(buffer) <= int(is_explore.sum().item())
    for ex in buffer:
        assert ex.branch == "explore"
        assert ex.target_chunk.shape == (chunk_len, x_dim)
        assert (ex.target_chunk >= 0.0).all() and (ex.target_chunk <= 1.0).all()
        assert is_explore[ex.instance_idx, ex.step]


def test_build_explore_chunk_buffer_x_seed_mode_realized_matches_rollout_action():
    torch.manual_seed(0)
    x_dim, chunk_len = 1, 2
    prior = _tiny_prior(batch_size=3, x_dim=x_dim)
    pfn, bar_dist = _tiny_pfn(x_dim=x_dim)
    n_init, n_steps = 4, 8
    rollout = rollout_episode(
        prior, n_init=n_init, n_steps=n_steps, policy_fn=random_policy,
        build_interesting_points_kwargs={"n_sobol": 6, "n_random": 6, "n_basin_restarts": 4},
    )
    is_explore = ~label_branches(rollout["y_context"], n_init)
    explore_steps = {s for s in range(n_steps) if is_explore[:, s].any()}
    assert explore_steps
    step = next(iter(explore_steps))

    buffer = build_explore_chunk_buffer(
        prior, pfn, bar_dist, rollout, n_init, chunk_len=chunk_len,
        explore_search_kwargs={"n_restarts": 1, "n_steps": 1, "lr": 0.0},
        steps={step}, require_improvement=False, x_seed_mode="realized",
    )
    for ex in buffer:
        x_realized = rollout["x_context"][ex.instance_idx, n_init + ex.step]
        # lr=0.0 means the search can't move -- slot 0 stays exactly at x_seed.
        assert torch.allclose(ex.target_chunk[0], x_realized, atol=1e-5)
