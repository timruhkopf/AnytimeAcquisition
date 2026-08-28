import torch

from anytimeacquisition.models.action_head import AUX_FEATURE_NAMES, ActionHead, pfn_dims
from anytimeacquisition.models.pfn import PFN


def _build(x_dim=2, d_model=16, n_layers=2, n_heads=2, batch_size=3, n_train=5):
    torch.manual_seed(0)
    pfn = PFN(x_dim=x_dim, d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=32, n_bins=16)
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
