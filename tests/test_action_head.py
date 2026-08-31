import torch

from anytimeacquisition.models.action_head import (
    AUX_FEATURE_NAMES,
    ActionHead,
    action_head_policy_fn,
    build_rollout_aux_features,
    pfn_dims,
)
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.trainer.exit_rollout import random_policy, rollout_episode


def _build(x_dim=2, d_model=16, n_layers=2, n_heads=2, batch_size=3, n_train=5):
    torch.manual_seed(0)
    pfn = PFN(max_x_dim=x_dim, d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=32, n_bins=16)
    pfn_d_model, pfn_n_layers = pfn_dims(pfn)
    action_head = ActionHead(pfn_d_model=pfn_d_model, pfn_n_layers=pfn_n_layers, x_dim=x_dim, d_model=16, n_heads=2, d_ff=32)
    x_train = torch.rand(batch_size, n_train, x_dim)
    y_train = torch.rand(batch_size, n_train)
    aux_features = {name: torch.rand(batch_size) for name in AUX_FEATURE_NAMES}
    return pfn, action_head, x_train, y_train, aux_features


def test_forward_pass_runs_at_target_shapes():
    x_dim, batch_size = 3, 4
    pfn, action_head, x_train, y_train, aux_features = _build(x_dim=x_dim, batch_size=batch_size)
    out = action_head(pfn, x_train, y_train, aux_features)
    assert out["alpha"].shape == (batch_size, x_dim)
    assert out["beta"].shape == (batch_size, x_dim)
    assert out["value"].shape == (batch_size,)


def test_beta_params_stay_at_or_above_one():
    # The other_diff prototype found NaN gradients when alpha/beta < 1 --
    # the softplus(x) + 1.0 clamp must hold regardless of raw head output.
    pfn, action_head, x_train, y_train, aux_features = _build()
    out = action_head(pfn, x_train, y_train, aux_features)
    assert (out["alpha"] >= 1.0).all()
    assert (out["beta"] >= 1.0).all()


def test_pfn_gradients_are_none_after_action_head_backward():
    pfn, action_head, x_train, y_train, aux_features = _build()
    out = action_head(pfn, x_train, y_train, aux_features)
    loss = out["alpha"].sum() + out["beta"].sum() + out["value"].sum()
    loss.backward()
    assert all(p.grad is None for p in pfn.parameters())
    assert all(p.grad is not None for p in action_head.parameters())


def test_action_head_output_is_batch_independent():
    # Each batch element's action distribution must depend only on its own
    # context, not leak across the batch dimension (a real bug class for
    # anything with batched attention -- cheap and worth checking directly).
    pfn, action_head, x_train, y_train, aux_features = _build(batch_size=3)
    out = action_head(pfn, x_train, y_train, aux_features)

    x_train_pert = x_train.clone()
    x_train_pert[0] += 1.0
    aux_pert = {k: v.clone() for k, v in aux_features.items()}
    out_pert = action_head(pfn, x_train_pert, y_train, aux_pert)

    assert torch.allclose(out["alpha"][1:], out_pert["alpha"][1:])
    assert torch.allclose(out["beta"][1:], out_pert["beta"][1:])
    assert torch.allclose(out["value"][1:], out_pert["value"][1:])


def test_build_rollout_aux_features_matches_expected_shapes_and_ranges():
    torch.manual_seed(0)
    n_init, n_steps, batch_size = 4, 10, 3
    prior = BNNPrior(batch_size=batch_size, x_dim=2, seed=0)
    rollout = rollout_episode(prior, n_init=n_init, n_steps=n_steps, policy_fn=random_policy)

    aux0 = build_rollout_aux_features(rollout, step=0, n_steps=n_steps)
    assert set(aux0) == set(AUX_FEATURE_NAMES)
    assert aux0["step_count"].shape == (batch_size,)
    assert (aux0["step_count"] == 0.0).all()
    assert torch.allclose(aux0["remaining_budget"], torch.full((batch_size,), 1.0))
    # No history yet (step < trend_window default of 3) -> trend is exactly 0.
    assert (aux0["improvement_trend"] == 0.0).all()

    aux_last = build_rollout_aux_features(rollout, step=n_steps - 1, n_steps=n_steps)
    assert (aux_last["step_count"] == n_steps - 1).all()
    assert torch.allclose(aux_last["remaining_budget"], torch.full((batch_size,), 1.0 / n_steps))
    assert (aux_last["improvement_trend"] >= 0.0).all()  # max(0, ...) clamp


def test_action_head_policy_fn_step_counter_advances_and_shape_is_correct():
    x_dim, batch_size = 2, 3
    pfn = PFN(max_x_dim=x_dim, d_model=16, n_heads=2, n_layers=2, d_ff=32, n_bins=16)
    pfn_d_model, pfn_n_layers = pfn_dims(pfn)
    torch.manual_seed(0)
    action_head = ActionHead(pfn_d_model=pfn_d_model, pfn_n_layers=pfn_n_layers, x_dim=x_dim, d_model=16, n_heads=2, d_ff=32)

    n_init, n_steps = 4, 6
    prior = BNNPrior(batch_size=batch_size, x_dim=x_dim, seed=0)
    policy_fn = action_head_policy_fn(action_head, pfn, n_steps=n_steps, sample=False)
    rollout = rollout_episode(prior, n_init=n_init, n_steps=n_steps, policy_fn=policy_fn)

    assert rollout["x_context"].shape == (batch_size, n_init + n_steps, x_dim)
    assert (rollout["x_context"] >= 0.0).all() and (rollout["x_context"] <= 1.0).all()


def test_action_head_policy_fn_sample_true_returns_valid_actions():
    x_dim, batch_size = 1, 4
    pfn = PFN(max_x_dim=x_dim, d_model=16, n_heads=2, n_layers=1, d_ff=32, n_bins=16)
    pfn_d_model, pfn_n_layers = pfn_dims(pfn)
    torch.manual_seed(0)
    action_head = ActionHead(pfn_d_model=pfn_d_model, pfn_n_layers=pfn_n_layers, x_dim=x_dim, d_model=16, n_heads=2, d_ff=32)

    x_train, y_train = torch.rand(batch_size, 5, x_dim), torch.rand(batch_size, 5)
    policy_fn = action_head_policy_fn(action_head, pfn, n_steps=5, sample=True)
    action = policy_fn(x_train, y_train, x_dim)
    assert action.shape == (batch_size, x_dim)
    assert (action >= 0.0).all() and (action <= 1.0).all()
