import pytest
import torch

from anytimeacquisition.callbacks.handler import Callback
from anytimeacquisition.models.action_head import ActionHead, pfn_dims
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.trainer.action_head_imitation_trainer import ActionHeadImitationTrainer


def _build(x_dim=1, batch_size=4):
    torch.manual_seed(0)
    pfn = PFN(max_x_dim=x_dim, d_model=16, n_heads=2, n_layers=1, d_ff=32, n_bins=16)
    pfn.eval()
    pfn_d_model, pfn_n_layers = pfn_dims(pfn)
    action_head = ActionHead(pfn_d_model=pfn_d_model, pfn_n_layers=pfn_n_layers, x_dim=x_dim, d_model=16, n_heads=2, d_ff=32)
    prior = BNNPrior(batch_size=batch_size, x_dim=x_dim, seed=1)
    return pfn, pfn.bar_dist, prior, action_head


@pytest.mark.parametrize("branches", [["exploit"], ["explore"], ["exploit", "explore"]])
def test_trainer_runs_end_to_end_for_each_branch_setting(branches):
    pfn, bar_dist, prior, action_head = _build()
    build_ip_kwargs = {"n_sobol": 4, "n_random": 4, "n_basin_restarts": 2} if "explore" in branches else None

    trainer = ActionHeadImitationTrainer(
        pfn=pfn, bar_dist=bar_dist, prior=prior, action_head=action_head, branches=branches,
        n_rollouts=6, n_init=3, n_steps=5, log_every=1,
        exploit_search_kwargs={"n_restarts": 2, "n_steps": 5},
        explore_search_kwargs={"n_restarts": 2, "n_steps": 5},
        build_interesting_points_kwargs=build_ip_kwargs,
    )
    result = trainer.run()
    history = result["history"]

    assert len(history["policy_nll/train"]) == 6
    total_exploit = sum(history["n_examples/exploit"])
    total_explore = sum(history["n_examples/explore"])
    if "exploit" in branches:
        assert total_exploit > 0
    else:
        assert total_exploit == 0
    if "explore" in branches:
        assert total_explore > 0
    else:
        assert total_explore == 0


def test_loss_decreases_over_a_short_integrated_run():
    torch.manual_seed(0)
    pfn, bar_dist, prior, action_head = _build(batch_size=8)
    trainer = ActionHeadImitationTrainer(
        pfn=pfn, bar_dist=bar_dist, prior=prior, action_head=action_head, branches=["exploit"],
        n_rollouts=40, n_init=3, n_steps=6, log_every=1, lr=3e-3,
        exploit_search_kwargs={"n_restarts": 4, "n_steps": 10},
    )
    history = trainer.run()["history"]
    losses = [v for v in history["policy_nll/train"] if v == v]  # filter any nan (empty rollouts)
    assert len(losses) >= 2
    # Mean of the second half should be lower than the first half -- a
    # single-step comparison is too noisy (rollout-to-rollout example counts
    # vary), matching PFNTrainer's own smoke-test spirit rather than its
    # exact first-vs-last-point comparison.
    half = len(losses) // 2
    assert sum(losses[half:]) / len(losses[half:]) < sum(losses[:half]) / len(losses[:half])


def test_callback_metrics_show_up_in_history_and_on_log():
    pfn, bar_dist, prior, action_head = _build()
    logged = []
    trainer = ActionHeadImitationTrainer(
        pfn=pfn, bar_dist=bar_dist, prior=prior, action_head=action_head, branches=["exploit"],
        n_rollouts=3, n_init=3, n_steps=4, log_every=1,
        exploit_search_kwargs={"n_restarts": 2, "n_steps": 5},
        on_log=lambda step, metrics: logged.append((step, metrics)),
        callbacks=[Callback(name="edge_case", fn=lambda step, t: {"flag": 1.0})],
    )
    result = trainer.run()

    assert result["history"]["edge_case/flag"] == [1.0, 1.0, 1.0]
    assert all("edge_case/flag" in metrics for _, metrics in logged)


def test_branches_validation_rejects_unknown_branch():
    pfn, bar_dist, prior, action_head = _build()
    with pytest.raises(AssertionError):
        ActionHeadImitationTrainer(
            pfn=pfn, bar_dist=bar_dist, prior=prior, action_head=action_head, branches=["nonsense"],
        )


def test_explore_branch_requires_build_interesting_points_kwargs():
    pfn, bar_dist, prior, action_head = _build()
    with pytest.raises(AssertionError):
        ActionHeadImitationTrainer(
            pfn=pfn, bar_dist=bar_dist, prior=prior, action_head=action_head, branches=["explore"],
        )


def test_dagger_beta_decays_and_is_logged():
    pfn, bar_dist, prior, action_head = _build()
    trainer = ActionHeadImitationTrainer(
        pfn=pfn, bar_dist=bar_dist, prior=prior, action_head=action_head, branches=["exploit"],
        n_rollouts=6, n_init=3, n_steps=4, log_every=1,
        exploit_search_kwargs={"n_restarts": 2, "n_steps": 5},
        dagger_decay_rounds=4, dagger_beta_min=0.1,
    )
    history = trainer.run()["history"]
    betas = history["dagger/beta"]
    assert betas[0] == 1.0
    assert betas == sorted(betas, reverse=True), "beta must be non-increasing over rollouts"
    assert all(b >= 0.1 for b in betas), "beta must never drop below dagger_beta_min"
    assert betas[-1] == pytest.approx(0.1)


def test_without_dagger_beta_is_always_one():
    # dagger_decay_rounds=None explicitly -- the default is "auto" (mixing
    # on, phased in) since 2026-09-01; this test is specifically about the
    # opt-out path, not the default.
    pfn, bar_dist, prior, action_head = _build()
    trainer = ActionHeadImitationTrainer(
        pfn=pfn, bar_dist=bar_dist, prior=prior, action_head=action_head, branches=["exploit"],
        n_rollouts=3, n_init=3, n_steps=4, log_every=1,
        exploit_search_kwargs={"n_restarts": 2, "n_steps": 5},
        dagger_decay_rounds=None,
    )
    history = trainer.run()["history"]
    assert history["dagger/beta"] == [1.0, 1.0, 1.0]


def test_checkpoint_round_trip(tmp_path):
    pfn, bar_dist, prior, action_head = _build()
    checkpoint_path = tmp_path / "action_head_ckpt.pt"
    model_config = {"pfn_d_model": pfn_dims(pfn)[0], "pfn_n_layers": pfn_dims(pfn)[1], "x_dim": 1, "d_model": 16, "n_heads": 2, "d_ff": 32, "dropout": 0.0}
    trainer = ActionHeadImitationTrainer(
        pfn=pfn, bar_dist=bar_dist, prior=prior, action_head=action_head, branches=["exploit"],
        n_rollouts=2, n_init=3, n_steps=4, log_every=1,
        exploit_search_kwargs={"n_restarts": 2, "n_steps": 5},
        checkpoint_path=checkpoint_path, model_config=model_config,
        extra_checkpoint_metadata={"mlflow_run_id": "abc123"},
    )
    trainer.run()

    ckpt = torch.load(checkpoint_path, weights_only=False)
    assert ckpt["config"] == model_config
    assert ckpt["mlflow_run_id"] == "abc123"
    reloaded = ActionHead(**model_config)
    reloaded.load_state_dict(ckpt["model_state"])

    from anytimeacquisition.pipelines.action_head_imitation import load_action_head_checkpoint

    loaded_action_head, loaded_ckpt = load_action_head_checkpoint(checkpoint_path)
    assert loaded_ckpt["mlflow_run_id"] == "abc123"
    for p1, p2 in zip(action_head.parameters(), loaded_action_head.parameters()):
        assert torch.equal(p1, p2)


def test_new_diagnostic_metrics_are_present_and_finite():
    pfn, bar_dist, prior, action_head = _build(batch_size=8)
    trainer = ActionHeadImitationTrainer(
        pfn=pfn, bar_dist=bar_dist, prior=prior, action_head=action_head, branches=["exploit", "explore"],
        n_rollouts=3, n_init=3, n_steps=6, log_every=1,
        exploit_search_kwargs={"n_restarts": 2, "n_steps": 5},
        explore_search_kwargs={"n_restarts": 1, "n_steps": 5},
        build_interesting_points_kwargs={"n_sobol": 4, "n_random": 4, "n_basin_restarts": 2},
        dagger_decay_rounds=3, dagger_beta_min=0.1,
    )
    history = trainer.run()["history"]

    for key in ("policy/beta_entropy", "grad_norm/action_head", "exploit/target_distance",
                "explore/signal_rate_train", "dagger/frac_self_generated", "explore/weighted_nll_reduction"):
        assert key in history, f"missing metric {key}"
        assert all(v == v for v in history[key]), f"{key} produced a NaN"  # v==v is False for NaN


def test_max_explore_steps_per_rollout_limits_distinct_explore_steps():
    pfn, bar_dist, prior, action_head = _build(batch_size=8)
    trainer = ActionHeadImitationTrainer(
        pfn=pfn, bar_dist=bar_dist, prior=prior, action_head=action_head, branches=["explore"],
        n_rollouts=1, n_init=3, n_steps=10, log_every=1,
        explore_search_kwargs={"n_restarts": 1, "n_steps": 5},
        build_interesting_points_kwargs={"n_sobol": 4, "n_random": 4, "n_basin_restarts": 2},
        max_explore_steps_per_rollout=2,
    )
    from anytimeacquisition.trainer.exit_rollout import random_policy, rollout_episode

    torch.manual_seed(trainer.seed)
    rollout = rollout_episode(
        prior, n_init=3, n_steps=10, policy_fn=random_policy,
        build_interesting_points_kwargs={"n_sobol": 4, "n_random": 4, "n_basin_restarts": 2},
    )
    examples, extra = trainer._collect_examples(rollout)
    distinct_steps = {ex.step for ex in examples}
    assert len(distinct_steps) <= 2
    assert "explore/signal_rate_train" in extra


def test_fill_unselected_explore_steps_with_exploit_adds_filler_examples():
    pfn, bar_dist, prior, action_head = _build(batch_size=8)
    build_ip_kwargs = {"n_sobol": 4, "n_random": 4, "n_basin_restarts": 2}

    trainer_no_fill = ActionHeadImitationTrainer(
        pfn=pfn, bar_dist=bar_dist, prior=prior, action_head=action_head, branches=["exploit", "explore"],
        n_rollouts=1, n_init=3, n_steps=10, log_every=1,
        exploit_search_kwargs={"n_restarts": 2, "n_steps": 5}, explore_search_kwargs={"n_restarts": 1, "n_steps": 5},
        build_interesting_points_kwargs=build_ip_kwargs, max_explore_steps_per_rollout=2,
    )
    trainer_fill = ActionHeadImitationTrainer(
        pfn=pfn, bar_dist=bar_dist, prior=BNNPrior(batch_size=8, x_dim=1, seed=1), action_head=action_head,
        branches=["exploit", "explore"],
        n_rollouts=1, n_init=3, n_steps=10, log_every=1,
        exploit_search_kwargs={"n_restarts": 2, "n_steps": 5}, explore_search_kwargs={"n_restarts": 1, "n_steps": 5},
        build_interesting_points_kwargs=build_ip_kwargs, max_explore_steps_per_rollout=2,
        fill_unselected_explore_steps_with_exploit=True,
    )
    from anytimeacquisition.trainer.exit_rollout import random_policy, rollout_episode

    torch.manual_seed(0)
    rollout = rollout_episode(
        trainer_no_fill.prior, n_init=3, n_steps=10, policy_fn=random_policy,
        build_interesting_points_kwargs=build_ip_kwargs,
    )
    examples_no_fill, extra_no_fill = trainer_no_fill._collect_examples(rollout)

    torch.manual_seed(0)
    rollout2 = rollout_episode(
        trainer_fill.prior, n_init=3, n_steps=10, policy_fn=random_policy,
        build_interesting_points_kwargs=build_ip_kwargs,
    )
    examples_fill, extra_fill = trainer_fill._collect_examples(rollout2)

    assert "n_examples/exploit_filler" not in extra_no_fill
    assert extra_fill.get("n_examples/exploit_filler", 0) > 0
    assert len(examples_fill) > len(examples_no_fill)
