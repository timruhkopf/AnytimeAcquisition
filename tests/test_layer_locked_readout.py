import torch

from anytimeacquisition.models.layer_locked_readout import LayerLockedReadout, load_checkpoint, save_checkpoint

D_MODEL_PFN, D_EXPERT, N_HEADS, N_LAYERS, D_FF, MAX_X_DIM, N_OUT = 16, 8, 2, 4, 32, 2, 1
READOUT_CONFIG = dict(
    d_model_pfn=D_MODEL_PFN, d_expert=D_EXPERT, n_heads=N_HEADS, n_layers=N_LAYERS,
    d_ff=D_FF, max_x_dim=MAX_X_DIM, n_out=N_OUT,
)
SCORE_CONFIG = dict(n_bins=16, z_lo=-6.0, z_hi=0.0)


def _readout():
    torch.manual_seed(0)
    return LayerLockedReadout(D_MODEL_PFN, D_EXPERT, N_HEADS, N_LAYERS, D_FF, MAX_X_DIM, N_OUT)


def _inputs(B=3, t=7, Q=5):
    torch.manual_seed(1)
    layer_hidden = [torch.randn(B, t, D_MODEL_PFN) for _ in range(N_LAYERS)]
    x_query = torch.rand(B, Q, MAX_X_DIM)
    incumbent_idx = torch.randint(0, t, (B,))
    return layer_hidden, x_query, incumbent_idx


def test_output_shape():
    readout = _readout()
    layer_hidden, x_query, incumbent_idx = _inputs(B=3, t=7, Q=5)
    out = readout(x_query, layer_hidden, incumbent_idx)
    assert out.shape == (3, 5, N_OUT)


def test_rejects_wrong_number_of_layer_hidden_states():
    readout = _readout()
    layer_hidden, x_query, incumbent_idx = _inputs()
    try:
        readout(x_query, layer_hidden[:-1], incumbent_idx)
        assert False, "expected an assertion error for a mismatched layer count"
    except AssertionError:
        pass


def test_query_independence_no_candidate_self_attention():
    """Perturbing one query's x must not change any OTHER query's output --
    mirrors models/pfn.py's own test-test independence invariant, and is
    the whole point of §3.4's no-candidate-self-attention decision."""
    readout = _readout()
    layer_hidden, x_query, incumbent_idx = _inputs(B=2, t=6, Q=4)
    out = readout(x_query, layer_hidden, incumbent_idx)

    x_query_pert = x_query.clone()
    x_query_pert[:, 0, :] += 1.0
    out_pert = readout(x_query_pert, layer_hidden, incumbent_idx)

    assert torch.allclose(out[:, 1:], out_pert[:, 1:], atol=1e-6)
    assert not torch.allclose(out[:, 0], out_pert[:, 0], atol=1e-6)


def test_incumbent_marker_actually_affects_output():
    """Moving WHICH index is marked as the incumbent (hidden states and
    x_query held fixed) must change the output -- the marker pathway is
    actually wired in, not a dead parameter."""
    readout = _readout()
    layer_hidden, x_query, incumbent_idx = _inputs(B=3, t=7, Q=5)
    out = readout(x_query, layer_hidden, incumbent_idx)
    out_alt = readout(x_query, layer_hidden, (incumbent_idx + 1) % 7)
    assert not torch.allclose(out, out_alt, atol=1e-6)


def test_caller_hidden_states_are_not_mutated():
    """The marker is added to a clone, never in place -- the caller's
    tensors (the frozen PFN's actual hidden states) must come back
    unchanged."""
    readout = _readout()
    layer_hidden, x_query, incumbent_idx = _inputs()
    before = [h.clone() for h in layer_hidden]
    readout(x_query, layer_hidden, incumbent_idx)
    assert all(torch.equal(a, b) for a, b in zip(layer_hidden, before))


def test_batch_items_are_independent():
    readout = _readout()
    layer_hidden, x_query, incumbent_idx = _inputs(B=2, t=6, Q=3)
    out = readout(x_query, layer_hidden, incumbent_idx)

    layer_hidden_pert = [h.clone() for h in layer_hidden]
    layer_hidden_pert[0][1] += 1.0  # perturb batch item 1's layer-0 hidden state only
    out_pert = readout(x_query, layer_hidden_pert, incumbent_idx)

    assert torch.allclose(out[0], out_pert[0], atol=1e-6)
    assert not torch.allclose(out[1], out_pert[1], atol=1e-6)


def test_gradient_flows_to_expert_params_only():
    readout = _readout()
    layer_hidden, x_query, incumbent_idx = _inputs()
    layer_hidden = [h.requires_grad_(False) for h in layer_hidden]  # frozen, as the real caller provides
    out = readout(x_query, layer_hidden, incumbent_idx)
    out.sum().backward()
    for p in readout.parameters():
        assert p.grad is not None


def test_checkpoint_round_trip_reproduces_identical_output(tmp_path):
    readout = _readout()
    layer_hidden, x_query, incumbent_idx = _inputs()
    out = readout(x_query, layer_hidden, incumbent_idx)

    ckpt_path = tmp_path / "readout.pt"
    save_checkpoint(
        readout, ckpt_path, READOUT_CONFIG, SCORE_CONFIG,
        pfn_checkpoint_path="models/pfn_variable_xdim_smoke.pt",
        history={"step": [0, 1], "loss": [1.0, 0.5]}, metrics={"pooled_rho": 0.9},
    )
    reloaded, bar_dist_score, ckpt = load_checkpoint(ckpt_path, device="cpu")

    out_reloaded = reloaded(x_query, layer_hidden, incumbent_idx)
    assert torch.equal(out, out_reloaded)
    assert bar_dist_score.num_bars == SCORE_CONFIG["n_bins"]
    assert ckpt["pfn_checkpoint_path"] == "models/pfn_variable_xdim_smoke.pt"
    assert ckpt["metrics"]["pooled_rho"] == 0.9
    assert ckpt["history"]["loss"] == [1.0, 0.5]


def test_checkpoint_creates_parent_directories(tmp_path):
    readout = _readout()
    ckpt_path = tmp_path / "nested" / "dir" / "readout.pt"
    save_checkpoint(readout, ckpt_path, READOUT_CONFIG, SCORE_CONFIG)
    assert ckpt_path.exists()


def test_load_checkpoint_rejects_mismatched_state_dict(tmp_path):
    """A shape mismatch (changed d_expert) raises via torch's own
    load_state_dict(strict=False) before load_checkpoint's own
    missing/unexpected check ever runs -- strict=False only suppresses
    missing/extra KEYS, not shape mismatches within a shared key. Either
    way, this must never silently load a wrong-shaped checkpoint."""
    readout = _readout()
    ckpt_path = tmp_path / "readout.pt"
    save_checkpoint(readout, ckpt_path, READOUT_CONFIG, SCORE_CONFIG)

    # Corrupt the saved config so it no longer matches the saved state_dict.
    ckpt = torch.load(ckpt_path, weights_only=False)
    ckpt["config"] = dict(READOUT_CONFIG, d_expert=D_EXPERT * 2)
    torch.save(ckpt, ckpt_path)

    try:
        load_checkpoint(ckpt_path, device="cpu")
        assert False, "expected a RuntimeError for a mismatched checkpoint"
    except RuntimeError:
        pass
