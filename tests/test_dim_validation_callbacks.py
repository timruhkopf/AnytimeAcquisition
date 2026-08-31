import pytest
import torch

from anytimeacquisition.callbacks.dim_validation import build_dim_validation_callbacks
from anytimeacquisition.callbacks.handler import CallbackHandler
from anytimeacquisition.models.pfn import PFN

FAST_ECDF_KWARGS = dict(ecdf_n_draws=3, ecdf_samples_per_draw=20, ecdf_n_samples=50)


class _FakeTrainer:
    def __init__(self, model):
        self.model = model
        self.bar_dist = model.bar_dist


def _make_trainer(max_x_dim=4):
    torch.manual_seed(0)
    model = PFN(max_x_dim=max_x_dim, d_model=16, n_heads=2, n_layers=1, d_ff=32, n_bins=16)
    return _FakeTrainer(model)


def test_reports_one_namespaced_metric_pair_per_dim():
    trainer = _make_trainer(max_x_dim=4)
    callbacks = build_dim_validation_callbacks(
        dims=[1, 2, 4], max_x_dim=4, n_val_context=4, n_val_points=10, ecdf_kwargs=FAST_ECDF_KWARGS,
    )
    handler = CallbackHandler(callbacks)

    metrics = handler.run(step=0, trainer=trainer, default_every_n_steps=1)

    assert set(metrics) == {
        "val_dim1/nll", "val_dim1/eval_mse",
        "val_dim2/nll", "val_dim2/eval_mse",
        "val_dim4/nll", "val_dim4/eval_mse",
    }
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


def test_rejects_dim_exceeding_max_x_dim():
    with pytest.raises(ValueError, match="exceed max_x_dim"):
        build_dim_validation_callbacks(dims=[1, 5], max_x_dim=4, n_val_context=4, ecdf_kwargs=FAST_ECDF_KWARGS)


def test_rejects_reserved_prior_kwargs():
    with pytest.raises(ValueError, match="x_dim"):
        build_dim_validation_callbacks(
            dims=[1], max_x_dim=4, prior_kwargs={"x_dim": 1}, ecdf_kwargs=FAST_ECDF_KWARGS,
        )


def test_dedicated_priors_share_fixed_architecture_across_probes():
    """reset() is called exactly once (at construction) per dedicated
    prior -- repeated probes sample fresh points from the SAME underlying
    random architecture, not a fresh one each time. Verified indirectly:
    the prior's own depth (fixed at reset()) doesn't change across calls."""
    callbacks = build_dim_validation_callbacks(
        dims=[2], max_x_dim=4, n_val_context=4, n_val_points=10, ecdf_kwargs=FAST_ECDF_KWARGS,
    )
    val_prior = callbacks[0].fn.__defaults__[0]  # the closure's captured val_prior default arg
    depth_before = val_prior.depth.clone()

    trainer = _make_trainer(max_x_dim=4)
    callbacks[0].fn(0, trainer)
    callbacks[0].fn(1, trainer)

    assert torch.equal(val_prior.depth, depth_before)


def test_custom_every_n_steps_overrides_default_cadence():
    callbacks = build_dim_validation_callbacks(
        dims=[1], max_x_dim=4, n_val_context=4, n_val_points=10,
        every_n_steps=3, ecdf_kwargs=FAST_ECDF_KWARGS,
    )
    handler = CallbackHandler(callbacks)
    trainer = _make_trainer(max_x_dim=4)

    fired = [step for step in range(9) if handler.run(step, trainer, default_every_n_steps=1)]

    assert fired == [0, 3, 6]
