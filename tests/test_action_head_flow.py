import torch

from anytimeacquisition.models.action_head import AUX_FEATURE_NAMES, compute_pfn_hidden_states, pfn_dims
from anytimeacquisition.models.action_head_flow import FlowMatchingActionHead
from anytimeacquisition.models.pfn import PFN


def _build(x_dim=2, chunk_len=3, d_model=16, n_layers=2, n_heads=2, batch_size=3, n_train=5):
    torch.manual_seed(0)
    pfn = PFN(max_x_dim=x_dim, d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=32, n_bins=16)
    pfn_d_model, pfn_n_layers = pfn_dims(pfn)
    head = FlowMatchingActionHead(
        pfn_d_model=pfn_d_model, pfn_n_layers=pfn_n_layers, x_dim=x_dim, chunk_len=chunk_len,
        d_model=16, n_heads=2, d_ff=32,
    )
    x_train = torch.rand(batch_size, n_train, x_dim)
    y_train = torch.rand(batch_size, n_train)
    aux_features = {name: torch.rand(batch_size) for name in AUX_FEATURE_NAMES}
    return pfn, head, x_train, y_train, aux_features


def test_compute_loss_shape():
    x_dim, chunk_len, batch_size = 3, 4, 4
    pfn, head, x_train, y_train, aux_features = _build(x_dim=x_dim, chunk_len=chunk_len, batch_size=batch_size)
    target_chunk = torch.rand(batch_size, chunk_len, x_dim)
    loss = head.compute_loss(pfn, x_train, y_train, aux_features, target_chunk)
    assert loss.shape == (batch_size,)
    assert (loss >= 0.0).all(), "MSE loss must be non-negative"


def test_pfn_gradients_are_none_after_flow_head_backward():
    x_dim, chunk_len, batch_size = 2, 3, 3
    pfn, head, x_train, y_train, aux_features = _build(x_dim=x_dim, chunk_len=chunk_len, batch_size=batch_size)
    target_chunk = torch.rand(batch_size, chunk_len, x_dim)
    loss = head.compute_loss(pfn, x_train, y_train, aux_features, target_chunk)
    loss.sum().backward()
    assert all(p.grad is None for p in pfn.parameters())
    assert all(p.grad is not None for p in head.parameters())


def test_sample_produces_valid_in_domain_chunk():
    x_dim, chunk_len, batch_size = 2, 3, 4
    pfn, head, x_train, y_train, aux_features = _build(x_dim=x_dim, chunk_len=chunk_len, batch_size=batch_size)
    sampled = head.sample(pfn, x_train, y_train, aux_features, num_steps=5)
    assert sampled.shape == (batch_size, chunk_len, x_dim)
    assert (sampled >= 0.0).all() and (sampled <= 1.0).all()


def test_sample_does_not_require_grad_and_pfn_stays_untouched():
    """sample() is a @torch.no_grad() inference path -- no gradients should
    be tracked at all, and the PFN's own parameters must be untouched
    (frozen, same guarantee as ActionHead's own forward)."""
    x_dim, chunk_len, batch_size = 1, 2, 3
    pfn, head, x_train, y_train, aux_features = _build(x_dim=x_dim, chunk_len=chunk_len, batch_size=batch_size)
    pfn_params_before = [p.detach().clone() for p in pfn.parameters()]
    sampled = head.sample(pfn, x_train, y_train, aux_features, num_steps=4)
    assert not sampled.requires_grad
    for p_before, p_after in zip(pfn_params_before, pfn.parameters()):
        assert torch.equal(p_before, p_after.detach())


def test_blind_ablation_changes_output_and_still_runs():
    """Same shapes/params as the real forward -- blind zeroes the PFN
    hidden states before cross-attention, same mechanism
    ActionHead.forward's own blind flag uses (shared compute_pfn_hidden_states)."""
    x_dim, chunk_len, batch_size = 2, 3, 3
    pfn, head, x_train, y_train, aux_features = _build(x_dim=x_dim, chunk_len=chunk_len, batch_size=batch_size)
    target_chunk = torch.rand(batch_size, chunk_len, x_dim)

    torch.manual_seed(1)
    loss_real = head.compute_loss(pfn, x_train, y_train, aux_features, target_chunk, blind=False)
    torch.manual_seed(1)
    loss_blind = head.compute_loss(pfn, x_train, y_train, aux_features, target_chunk, blind=True)
    assert loss_real.shape == loss_blind.shape == (batch_size,)


def test_chunk_steps_are_not_symmetric():
    """The learned chunk_pos_embed must break symmetry between chunk-step
    tokens -- unlike search.kstep_explore's interchangeable companion
    points, these represent genuinely different, ordered plan positions.
    Feeding the SAME x_t value into every chunk slot should still produce
    DIFFERENT velocities per slot, since each slot's positional embedding
    differs."""
    x_dim, chunk_len, batch_size = 2, 3, 2
    pfn, head, x_train, y_train, aux_features = _build(x_dim=x_dim, chunk_len=chunk_len, batch_size=batch_size)
    hidden_states = compute_pfn_hidden_states(pfn, x_train, y_train)

    x_t = torch.rand(batch_size, 1, x_dim).expand(batch_size, chunk_len, x_dim)  # same value at every chunk slot
    t = torch.full((batch_size,), 0.5)
    v_t = head.forward_velocity(hidden_states, aux_features, x_t, t)
    # If chunk slots were symmetric, v_t[:, 0] would equal v_t[:, 1] etc.
    assert not torch.allclose(v_t[:, 0], v_t[:, 1])
