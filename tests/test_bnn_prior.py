import torch

from anytimeacquisition.priors.bnn import BNNPrior

FAST_ECDF_KWARGS = dict(ecdf_n_draws=10, ecdf_samples_per_draw=100, ecdf_n_samples=100)


def _make_prior(**kwargs):
    kwargs = {**FAST_ECDF_KWARGS, "cache_dir": None, **kwargs}
    return BNNPrior(batch_size=4, x_dim=3, seed=0, **kwargs)


def test_evaluate_output_range_is_unit_interval():
    prior = _make_prior()
    x = torch.rand(4, 200, 3)
    y = prior.evaluate(x)
    assert y.min() >= 0.0
    assert y.max() <= 1.0


def test_evaluate_is_differentiable_wrt_x():
    prior = _make_prior()
    x = torch.rand(4, 8, 3, requires_grad=True)
    y = prior.evaluate(x)
    y.sum().backward()
    assert x.grad is not None
    assert (x.grad.abs() > 0).any()


def test_batch_elements_are_independent_architectures():
    prior = _make_prior()
    x = torch.rand(4, 20, 3)
    y1 = prior.evaluate(x)
    prior.reset()
    y2 = prior.evaluate(x)
    assert not torch.allclose(y1, y2)


def test_sample_episode_shapes():
    prior = _make_prior()
    x_tr, y_tr, x_te, y_te = prior.sample_episode(n_train=10, n_test=5)
    assert x_tr.shape == (4, 10, 3)
    assert y_tr.shape == (4, 10)
    assert x_te.shape == (4, 5, 3)
    assert y_te.shape == (4, 5)


def test_noise_toggle():
    prior = _make_prior()
    x = torch.rand(4, 20, 3)

    y_det_1 = prior.evaluate(x, noise=False)
    y_det_2 = prior.evaluate(x, noise=False)
    assert torch.equal(y_det_1, y_det_2)  # same weights, no noise -> identical

    y_noisy_1 = prior.evaluate(x, noise=True)
    y_noisy_2 = prior.evaluate(x, noise=True)
    assert not torch.equal(y_noisy_1, y_noisy_2)  # fresh noise draw each call


def test_sparseness_zeros_some_hidden_weights():
    sparse = _make_prior(sparseness=0.145)
    dense = _make_prior(sparseness=0.0)
    assert (sparse.W_h == 0).float().mean().item() > 0.05
    # tolerance, not exact equality: an unlucky float32 randn draw landing on
    # exactly 0.0 has nonzero (if tiny) probability across millions of values
    assert (dense.W_h == 0).float().mean().item() < 1e-5


def test_spurious_dimensions_get_zero_gradient():
    prior = BNNPrior(
        batch_size=32, x_dim=4, seed=0, frac_relevant_features=0.5,
        cache_dir=None, **FAST_ECDF_KWARGS,
    )
    x = torch.rand(32, 10, 4, requires_grad=True)
    y = prior.evaluate(x, noise=False)
    y.sum().backward()

    irrelevant = prior.relevant_mask == 0
    assert irrelevant.any() and (~irrelevant).any()  # sanity: a real mix
    grad_per_dim = x.grad.abs().sum(dim=1)  # [B, d]
    assert (grad_per_dim[irrelevant] == 0).all()
    assert (grad_per_dim[~irrelevant] > 0).all()


def test_variable_dim_disabled_matches_full_dim():
    # Default: variable_dim_min=None -> every instance's active_dim is the
    # full x_dim, byte-identical to not having the feature at all.
    prior = _make_prior()
    assert torch.equal(prior.active_dim, torch.full((4,), 3))
    assert torch.equal(prior.active_dim_mask, torch.ones(4, 3))


def test_variable_dim_active_dim_in_range():
    prior = _make_prior(variable_dim_min=1)
    assert (prior.active_dim >= 1).all()
    assert (prior.active_dim <= 3).all()
    # Batch-uniform, not per-instance (2026-08-31, see
    # docs/log/2026-08-31-variable-xdim-training-stagnation.md): every
    # instance in ONE reset() shares the same active_dim.
    assert prior.active_dim.unique().numel() == 1


def test_variable_dim_active_dim_varies_step_to_step():
    prior = _make_prior(variable_dim_min=1)
    seen = set()
    for _ in range(20):
        prior.reset()
        seen.add(prior.active_dim[0].item())
    # sanity: with reset() resampling uniformly over {1,2,3} every step,
    # never seeing more than one value in 20 resets is overwhelmingly
    # unlikely -- this is the axis active_dim is now allowed to vary on.
    assert len(seen) > 1


def test_variable_dim_zeros_x_beyond_active_dim():
    prior = BNNPrior(
        batch_size=16, x_dim=4, seed=0, variable_dim_min=1,
        cache_dir=None, **FAST_ECDF_KWARGS,
    )
    x_tr, _, x_te, _ = prior.sample_episode(n_train=5, n_test=5)
    x = torch.cat([x_tr, x_te], dim=1)  # [B, 10, 4]

    inactive = prior.active_dim_mask == 0
    assert inactive.any() and (~inactive).any()  # sanity: a real mix
    for b in range(16):
        assert (x[b, :, inactive[b]] == 0).all()


def test_variable_dim_gets_zero_gradient_beyond_active_dim():
    prior = BNNPrior(
        batch_size=16, x_dim=4, seed=0, variable_dim_min=1, frac_relevant_features=1.0,
        cache_dir=None, **FAST_ECDF_KWARGS,
    )
    x = torch.rand(16, 10, 4, requires_grad=True)
    y = prior.evaluate(x, noise=False)
    y.sum().backward()

    inactive = prior.active_dim_mask == 0
    grad_per_dim = x.grad.abs().sum(dim=1)  # [B, d]
    assert (grad_per_dim[inactive] == 0).all()
    assert (grad_per_dim[~inactive] > 0).all()


def test_ecdf_cache_roundtrip(tmp_path):
    prior_a = BNNPrior(batch_size=4, x_dim=3, seed=0, cache_dir=tmp_path, **FAST_ECDF_KWARGS)
    cache_files = list(tmp_path.glob("*.pt"))
    assert len(cache_files) == 1

    # Different seed and batch size, same family config -> must load the
    # cached ECDF rather than re-fitting from scratch.
    prior_b = BNNPrior(batch_size=7, x_dim=3, seed=12345, cache_dir=tmp_path, **FAST_ECDF_KWARGS)

    assert torch.equal(prior_a.ecdf_sorted[0], prior_b.ecdf_sorted[0])
    assert prior_b.ecdf_sorted.shape[0] == 7
    assert len(list(tmp_path.glob("*.pt"))) == 1  # no second cache file written


def test_ecdf_sorted_override_skips_fitting_entirely():
    """callbacks/dim_validation.py's whole reason for existing: a prior
    constructed with a DIFFERENT x_dim can still share another prior's
    already-fit ecdf_sorted verbatim, rather than fitting its own (which
    would calibrate against only its own dimension's raw-output
    distribution -- see that module's docstring)."""
    source = BNNPrior(batch_size=4, x_dim=3, seed=0, cache_dir=None, **FAST_ECDF_KWARGS)

    reused = BNNPrior(
        batch_size=6, x_dim=1, seed=99, cache_dir=None, ecdf_sorted=source.ecdf_sorted,
    )

    assert torch.equal(reused.ecdf_sorted[0], source.ecdf_sorted[0])
    assert reused.ecdf_sorted.shape[0] == 6  # expanded to its own batch_size, not source's


def test_ecdf_sorted_override_accepts_single_row():
    source = BNNPrior(batch_size=4, x_dim=3, seed=0, cache_dir=None, **FAST_ECDF_KWARGS)

    reused = BNNPrior(batch_size=2, x_dim=1, seed=0, cache_dir=None, ecdf_sorted=source.ecdf_sorted[:1])

    assert torch.equal(reused.ecdf_sorted[0], source.ecdf_sorted[0])
