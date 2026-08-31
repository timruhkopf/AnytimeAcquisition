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
