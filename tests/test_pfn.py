import torch

from anytimeacquisition.models.pfn import PFN


def _make_model(**kwargs):
    kwargs = {"max_x_dim": 2, "d_model": 32, "n_heads": 4, "n_layers": 2, "d_ff": 64, "n_bins": 16, **kwargs}
    return PFN(**kwargs)


def test_output_shape():
    model = _make_model()
    x_train, y_train = torch.rand(3, 5, 2), torch.rand(3, 5)
    x_test = torch.rand(3, 4, 2)
    logits = model(x_train, y_train, x_test)
    assert logits.shape == (3, 4, 16)


def test_permutation_invariant_over_train_set():
    torch.manual_seed(0)
    model = _make_model()
    x_train, y_train = torch.rand(2, 6, 2), torch.rand(2, 6)
    x_test = torch.rand(2, 3, 2)

    logits = model(x_train, y_train, x_test)
    perm = torch.randperm(6)
    logits_perm = model(x_train[:, perm], y_train[:, perm], x_test)

    assert torch.allclose(logits, logits_perm, atol=1e-5)


def test_no_test_test_leakage():
    torch.manual_seed(0)
    model = _make_model()
    x_train, y_train = torch.rand(2, 6, 2), torch.rand(2, 6)
    x_test = torch.rand(2, 4, 2)

    logits = model(x_train, y_train, x_test)
    x_test_perturbed = x_test.clone()
    x_test_perturbed[:, 0, :] += 1.0
    logits_perturbed = model(x_train, y_train, x_test_perturbed)

    # perturbing test point 0 must not change any OTHER test point's logits
    assert torch.allclose(logits[:, 1:, :], logits_perturbed[:, 1:, :], atol=1e-5)
    # but it must change its own
    assert not torch.allclose(logits[:, 0, :], logits_perturbed[:, 0, :], atol=1e-5)


def test_gradient_isolated_between_test_points():
    torch.manual_seed(0)
    model = _make_model()
    x_train, y_train = torch.rand(1, 5, 2), torch.rand(1, 5)
    x_test = torch.rand(1, 3, 2, requires_grad=True)

    logits = model(x_train, y_train, x_test)
    logits[:, 0, :].sum().backward()

    # d(logits at test point 0) / d(x at test points 1, 2) must be exactly zero
    assert (x_test.grad[:, 1:, :] == 0).all()
    assert (x_test.grad[:, 0, :] != 0).any()


def test_return_hidden_states_shapes():
    model = _make_model(n_layers=3)
    x_train, y_train = torch.rand(2, 5, 2), torch.rand(2, 5)
    x_test = torch.rand(2, 3, 2)
    logits, hidden = model(x_train, y_train, x_test, return_hidden=True)
    assert len(hidden) == 3
    for h in hidden:
        assert h.shape == (2, 8, 32)  # Ntr + Nte, d_model


def test_train_key_padding_mask_ignores_padded_train_tokens():
    """A batch mixing episodes with different real n_train (padded to a
    common width, masked via train_key_padding_mask) must give each batch
    item exactly the logits it would get from an unpadded forward pass using
    only its real train points -- the padded rows must be fully invisible,
    not just numerically small (see pfn.py's module docstring for why this
    was added: ifBO's own train/test split-attention path doesn't support
    per-batch-item padding)."""
    torch.manual_seed(0)
    model = _make_model()
    B, Ntr, Nte = 2, 6, 3
    x_train, y_train = torch.rand(B, Ntr, 2), torch.rand(B, Ntr)
    x_test = torch.rand(B, Nte, 2)

    n_valid = 4  # item 1 only has 4 real train points, padded out to Ntr=6
    train_key_padding_mask = torch.ones(B, Ntr, dtype=torch.bool)
    train_key_padding_mask[1, n_valid:] = False

    logits_padded = model(x_train, y_train, x_test, train_key_padding_mask=train_key_padding_mask)
    logits_item0_ref = model(x_train[0:1], y_train[0:1], x_test[0:1])
    logits_item1_ref = model(x_train[1:2, :n_valid], y_train[1:2, :n_valid], x_test[1:2])

    assert torch.allclose(logits_padded[0:1], logits_item0_ref, atol=1e-5)
    assert torch.allclose(logits_padded[1:2], logits_item1_ref, atol=1e-5)


def test_n_features_batch_wide_default_matches_pfns4bo_convention():
    """With n_features omitted, a model built at max_x_dim wider than the
    actual x supplied must behave exactly like ifBO/PFNs4BO's own
    VariableNumFeaturesEncoder: rescale by max_x_dim/D then zero-pad D up to
    max_x_dim, applied uniformly across the whole batch. Sanity-checked here
    against a hand-computed equivalent forward pass through the same model,
    not just "doesn't crash"."""
    torch.manual_seed(0)
    model = _make_model(max_x_dim=4)
    B, Ntr, Nte, D = 2, 5, 3, 2
    x_train, y_train = torch.rand(B, Ntr, D), torch.rand(B, Ntr)
    x_test = torch.rand(B, Nte, D)

    logits = model(x_train, y_train, x_test)

    scale = model.max_x_dim / D
    x_train_padded = torch.cat([x_train * scale, torch.zeros(B, Ntr, model.max_x_dim - D)], dim=-1)
    x_test_padded = torch.cat([x_test * scale, torch.zeros(B, Nte, model.max_x_dim - D)], dim=-1)
    n_features_full = torch.full((B,), model.max_x_dim)
    logits_explicit = model(x_train_padded, y_train, x_test_padded, n_features=n_features_full)

    assert torch.allclose(logits, logits_explicit, atol=1e-5)


def test_n_features_ignores_garbage_beyond_real_count():
    """Per-instance n_features (BNNPrior's active_dim use case): content at
    a feature column index >= n_features[b] must be fully invisible, not
    just small -- _pad_and_rescale_features zeroes it regardless of what the
    caller put there, rather than trusting it's already zero."""
    torch.manual_seed(0)
    model = _make_model(max_x_dim=4)
    B, Ntr, Nte = 2, 5, 3
    x_train, y_train = torch.rand(B, Ntr, 4), torch.rand(B, Ntr)
    x_test = torch.rand(B, Nte, 4)
    n_features = torch.tensor([4, 2])  # item 1 only has 2 real dims of the 4
    x_train[1, :, 2:] = 0.0
    x_test[1, :, 2:] = 0.0

    logits = model(x_train, y_train, x_test, n_features=n_features)

    x_train_garbage = x_train.clone()
    x_train_garbage[1, :, 2:] = 99.0  # item 1's "padding" columns: garbage, not zero
    x_test_garbage = x_test.clone()
    x_test_garbage[1, :, 2:] = -99.0
    logits_garbage = model(x_train_garbage, y_train, x_test_garbage, n_features=n_features)

    assert torch.allclose(logits[1], logits_garbage[1], atol=1e-5)
    # item 0 (n_features == max_x_dim, nothing masked) is untouched by item 1's garbage
    assert torch.allclose(logits[0], logits_garbage[0], atol=1e-5)


def test_bar_dist_is_owned_submodule_not_a_trainable_parameter():
    model = _make_model(n_bins=16)
    assert model.bar_dist.num_bars == 16
    assert "bar_dist.borders" in model.state_dict()
    assert "bar_dist.bucket_widths" in model.state_dict()
    # buffers, not learnable weights -- must not show up in the optimizer's view
    assert not any(name.startswith("bar_dist.") for name, _ in model.named_parameters())
