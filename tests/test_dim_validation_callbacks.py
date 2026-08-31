import pytest
import torch

from anytimeacquisition.callbacks.dim_validation import build_dim_validation_callback
from anytimeacquisition.callbacks.handler import CallbackHandler
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.priors.bnn import BNNPrior

FAST_ECDF_KWARGS = dict(ecdf_n_draws=3, ecdf_samples_per_draw=20, ecdf_n_samples=50)


class _FakeTrainer:
    def __init__(self, model):
        self.model = model
        self.bar_dist = model.bar_dist


def _make_trainer(max_x_dim=4):
    torch.manual_seed(0)
    model = PFN(max_x_dim=max_x_dim, d_model=16, n_heads=2, n_layers=1, d_ff=32, n_bins=16)
    return _FakeTrainer(model)


def _shared_ecdf(x_dim=4, variable_dim_min=1):
    prior = BNNPrior(
        batch_size=4, x_dim=x_dim, variable_dim_min=variable_dim_min, seed=0,
        cache_dir=None, **FAST_ECDF_KWARGS,
    )
    return prior.ecdf_sorted


def _captured(fn, name):
    """Pull a variable captured by fn's closure -- white-box access to
    build_dim_validation_callback's internal val_priors dict, which isn't
    otherwise exposed (the Callback it returns only exposes `fn`)."""
    freevars = dict(zip(fn.__code__.co_freevars, (cell.cell_contents for cell in fn.__closure__)))
    return freevars[name]


def test_reports_metric_type_first_namespaced_keys():
    trainer = _make_trainer(max_x_dim=4)
    callback = build_dim_validation_callback(
        dims=[1, 2, 4], max_x_dim=4, ecdf_sorted=_shared_ecdf(),
        n_val_context=4, n_val_points=10,
    )
    handler = CallbackHandler([callback])

    metrics = handler.run(step=0, trainer=trainer, default_every_n_steps=1)

    assert set(metrics) == {
        "nll/val_dim1", "mse/val_dim1",
        "nll/val_dim2", "mse/val_dim2",
        "nll/val_dim4", "mse/val_dim4",
    }
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


def test_rejects_dim_exceeding_max_x_dim():
    with pytest.raises(ValueError, match="exceed max_x_dim"):
        build_dim_validation_callback(dims=[1, 5], max_x_dim=4, ecdf_sorted=_shared_ecdf(), n_val_context=4)


def test_rejects_reserved_prior_kwargs():
    with pytest.raises(ValueError, match="x_dim"):
        build_dim_validation_callback(
            dims=[1], max_x_dim=4, ecdf_sorted=_shared_ecdf(), prior_kwargs={"x_dim": 1},
        )


def test_dedicated_priors_share_the_passed_in_ecdf():
    shared = _shared_ecdf()
    callback = build_dim_validation_callback(
        dims=[1, 2], max_x_dim=4, ecdf_sorted=shared, n_val_context=4, n_val_points=10,
    )
    val_priors = _captured(callback.fn, "val_priors")

    for val_prior in val_priors.values():
        assert torch.equal(val_prior.ecdf_sorted[0], shared[0])


def test_dedicated_priors_keep_fixed_architecture_across_probes():
    """reset() is called exactly once (at construction) per dedicated
    prior -- repeated probes sample fresh points from the SAME underlying
    random architecture, not a fresh one each time."""
    shared = _shared_ecdf()
    callback = build_dim_validation_callback(
        dims=[2], max_x_dim=4, ecdf_sorted=shared, n_val_context=4, n_val_points=10,
    )
    val_priors = _captured(callback.fn, "val_priors")
    depth_before = val_priors[2].depth.clone()

    trainer = _make_trainer(max_x_dim=4)
    callback.fn(0, trainer)
    callback.fn(1, trainer)

    assert torch.equal(val_priors[2].depth, depth_before)


def test_custom_every_n_steps_overrides_default_cadence():
    callback = build_dim_validation_callback(
        dims=[1], max_x_dim=4, ecdf_sorted=_shared_ecdf(),
        n_val_context=4, n_val_points=10, every_n_steps=3,
    )
    handler = CallbackHandler([callback])
    trainer = _make_trainer(max_x_dim=4)

    fired = [step for step in range(9) if handler.run(step, trainer, default_every_n_steps=1)]

    assert fired == [0, 3, 6]
