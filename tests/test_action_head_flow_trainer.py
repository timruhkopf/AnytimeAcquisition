import pytest
import torch

from anytimeacquisition.models.action_head import pfn_dims
from anytimeacquisition.models.action_head_flow import FlowMatchingActionHead
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.trainer.action_head_flow_trainer import ActionHeadFlowTrainer


def _build(x_dim=1, chunk_len=3, batch_size=4):
    torch.manual_seed(0)
    pfn = PFN(max_x_dim=x_dim, d_model=16, n_heads=2, n_layers=1, d_ff=32, n_bins=16)
    pfn.eval()
    pfn_d_model, pfn_n_layers = pfn_dims(pfn)
    action_head = FlowMatchingActionHead(
        pfn_d_model=pfn_d_model, pfn_n_layers=pfn_n_layers, x_dim=x_dim, chunk_len=chunk_len,
        d_model=16, n_heads=2, d_ff=32,
    )
    prior = BNNPrior(batch_size=batch_size, x_dim=x_dim, seed=1)
    return pfn, pfn.bar_dist, prior, action_head


@pytest.mark.parametrize("exploit_on,explore_on", [(True, False), (False, True), (True, True)])
def test_trainer_runs_end_to_end_for_each_branch_setting(exploit_on, explore_on):
    chunk_len = 3
    pfn, bar_dist, prior, action_head = _build(chunk_len=chunk_len)
    exploit_search_kwargs = {"n_restarts": 2, "n_steps": 5} if exploit_on else None
    explore_search_kwargs = {"n_restarts": 2, "n_steps": 5} if explore_on else None
    build_ip_kwargs = {"n_sobol": 4, "n_random": 4, "n_basin_restarts": 2} if explore_on else None

    trainer = ActionHeadFlowTrainer(
        pfn=pfn, bar_dist=bar_dist, prior=prior, action_head=action_head,
        n_rollouts=4, n_init=3, n_steps=5, log_every=1, chunk_len=chunk_len,
        exploit_search_kwargs=exploit_search_kwargs,
        explore_search_kwargs=explore_search_kwargs,
        build_interesting_points_kwargs=build_ip_kwargs,
    )
    result = trainer.run()
    history = result["history"]

    assert len(history["policy_loss/train"]) == 4
    total_exploit = sum(history["n_examples/exploit"])
    total_explore = sum(history["n_examples/explore"])
    if exploit_on:
        assert total_exploit > 0
    else:
        assert total_exploit == 0
    if explore_on:
        assert total_explore > 0
    else:
        assert total_explore == 0


def test_chunk_len_mismatch_raises():
    chunk_len = 3
    pfn, bar_dist, prior, action_head = _build(chunk_len=chunk_len)
    with pytest.raises(AssertionError):
        ActionHeadFlowTrainer(
            pfn=pfn, bar_dist=bar_dist, prior=prior, action_head=action_head,
            n_rollouts=1, n_init=3, n_steps=5, chunk_len=chunk_len + 1,  # deliberately wrong
            exploit_search_kwargs={"n_restarts": 2, "n_steps": 5},
        )


def test_at_least_one_branch_must_be_on():
    pfn, bar_dist, prior, action_head = _build()
    with pytest.raises(AssertionError):
        ActionHeadFlowTrainer(
            pfn=pfn, bar_dist=bar_dist, prior=prior, action_head=action_head,
            n_rollouts=1, n_init=3, n_steps=5,
        )


def test_dagger_beta_decays_and_is_logged():
    chunk_len = 3
    pfn, bar_dist, prior, action_head = _build(chunk_len=chunk_len)
    trainer = ActionHeadFlowTrainer(
        pfn=pfn, bar_dist=bar_dist, prior=prior, action_head=action_head,
        n_rollouts=6, n_init=3, n_steps=5, log_every=1, chunk_len=chunk_len,
        exploit_search_kwargs={"n_restarts": 2, "n_steps": 5},
        dagger_decay_rounds=6, dagger_beta_min=0.1,
    )
    result = trainer.run()
    beta_history = result["history"]["dagger/beta"]
    assert beta_history[0] == pytest.approx(1.0)
    assert beta_history[-1] < beta_history[0]
    assert all(b >= 0.1 for b in beta_history)


def test_checkpoint_round_trip(tmp_path):
    chunk_len = 3
    pfn, bar_dist, prior, action_head = _build(chunk_len=chunk_len)
    ckpt_path = tmp_path / "flow_head.pt"
    trainer = ActionHeadFlowTrainer(
        pfn=pfn, bar_dist=bar_dist, prior=prior, action_head=action_head,
        n_rollouts=2, n_init=3, n_steps=5, log_every=1, chunk_len=chunk_len,
        exploit_search_kwargs={"n_restarts": 2, "n_steps": 5},
        checkpoint_path=ckpt_path, model_config={"chunk_len": chunk_len},
    )
    trainer.run()
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, weights_only=False)
    assert "model_state" in ckpt and ckpt["config"] == {"chunk_len": chunk_len}
