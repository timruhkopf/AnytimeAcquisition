import torch

from anytimeacquisition.models.pfn import PFN


def _make_model(**kwargs):
    kwargs = {"x_dim": 2, "d_model": 32, "n_heads": 4, "n_layers": 2, "d_ff": 64, "n_bins": 16, **kwargs}
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
