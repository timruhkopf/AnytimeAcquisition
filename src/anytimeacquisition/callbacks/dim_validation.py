"""Per-dimension validation callback (M2) -- built specifically to check
the hypothesis in `docs/log/2026-08-31-variable-xdim-training-stagnation.md`:
does a variable-x_dim PFN's NLL/eval_mse genuinely get worse at higher
dimensionality (curse-of-dimensionality, expected), independent of whatever
`BNNPrior.active_dim` happens to be doing on any given training step?

Each dimension in `dims` gets its own **dedicated, fixed-x_dim** `BNNPrior`
(`variable_dim_min=None`), built ONCE by `build_dim_validation_callback`
and reused for every subsequent probe -- not rebuilt per call. `reset()` is
called exactly once per dedicated prior (at construction), never again, so
every probe against a given dimension samples fresh points from the *same*
fixed underlying random architectures throughout the whole training run --
the way a held-out validation set should behave, not a fresh random task
each time (which would make the reported curve noisier and harder to read
trends from).

**All dedicated priors share one `ecdf_sorted`, passed in rather than each
independently fit.** `BNNPrior._fit_ecdf` pools raw outputs across many
`reset()` draws, which already spans the training prior's own dimension
range when `variable_dim_min` is set (each draw resamples `active_dim`) --
but a *separately constructed* fixed-`x_dim` prior fitting its own ECDF
would calibrate only against ITS OWN dimension's raw-output distribution,
independently whitening it to [0,1]. Do that per dimension and the
comparison across dims stops being apples-to-apples: each dimension's y
gets rescaled to fill [0,1] on its own terms, which can mask genuine
cross-dimension difficulty differences rather than reveal them. ifBO's own
reference implementation avoids exactly this by using one static,
precomputed calibration array regardless of which dimensionality
(`num_features`) a given sample happens to draw
(`ifbo/priors/ftpfn_prior.py`'s `output_sorted.npy`, loaded once, searched
via `np.searchsorted` for every sample no matter its own `num_params`) --
so here too, every dedicated validation prior (and, by the same logic,
the training prior itself) shares the ONE ECDF that was fit once, at the
start, from a distribution that already spans the sizes in question.

Requires the model being validated to already accept the largest `dim` in
`dims` (i.e. `max(dims) <= trainer.model.max_x_dim`) -- checked eagerly at
build time, not discovered later at the first probe.
"""
from typing import Any

import torch

from anytimeacquisition.callbacks.handler import Callback
from anytimeacquisition.priors.bnn import BNNPrior


def build_dim_validation_callback(
    dims: list[int],
    max_x_dim: int,
    ecdf_sorted: torch.Tensor,
    prior_kwargs: dict | None = None,
    n_val_context: int = 20,
    n_val_points: int = 200,
    seed: int = 0,
    every_n_steps: int | None = None,
) -> Callback:
    """One `Callback` covering every dimension in `dims`, reporting
    `nll/val_dimN` and `mse/val_dimN` for each -- metric-type first, so
    MLflow's dashboard groups all NLL curves together (train's own
    `nll/train` alongside every `nll/val_dimN`), all MSE curves together,
    rather than one folder per dimension with both metrics buried inside
    it. Deliberately one `Callback` (not one per dimension): each probe
    already does one forward pass per dimension computing both metrics
    from the same `logits`, so splitting into a `Callback` per metric type
    would mean two independent forward passes (and two independent
    `sample_episode` draws) per dimension for no benefit.

    `ecdf_sorted`: shared normalization reference, `[1, ecdf_n_samples]`
    or `[B, ecdf_n_samples]` -- typically the TRAINING prior's own
    `ecdf_sorted` (see module docstring for why every dedicated validation
    prior must share one, not fit its own). Required, not optional --
    silently defaulting to "each dimension fits its own" is exactly the
    bug this module exists to avoid.

    `prior_kwargs` should mirror the training run's own `priors.*` config
    (`depth_range`, `width_range`, `log_amp_range`, `sparseness`, ...) so
    the comparison is apples-to-apples against the same task family --
    only `x_dim`/`variable_dim_min`/`batch_size`/`ecdf_sorted` differ per
    dimension here and are set internally; passing any of those four
    raises, since silently overriding a caller's explicit value would be
    more confusing than refusing. `every_n_steps=None` (default) uses the
    trainer's own `log_every` cadence (see `callbacks/handler.py`).
    """
    reserved = {"x_dim", "variable_dim_min", "batch_size", "ecdf_sorted"}
    prior_kwargs = dict(prior_kwargs or {})
    clashing = reserved & prior_kwargs.keys()
    if clashing:
        raise ValueError(
            f"build_dim_validation_callback sets {sorted(reserved)} itself (per-dimension, or shared) -- "
            f"remove {sorted(clashing)} from prior_kwargs rather than overriding them here."
        )
    too_big = [d for d in dims if d > max_x_dim]
    if too_big:
        raise ValueError(f"dims {too_big} exceed max_x_dim={max_x_dim} -- the model can't accept them")

    val_priors = {
        d: BNNPrior(
            batch_size=n_val_context, x_dim=d, variable_dim_min=None, seed=seed,
            ecdf_sorted=ecdf_sorted, **prior_kwargs,
        )
        for d in dims
    }

    def probe(step: int, trainer: Any) -> dict:
        metrics = {}
        for d, val_prior in val_priors.items():
            x_tr, y_tr, x_te, y_te = val_prior.sample_episode(n_train=n_val_context, n_test=n_val_points)
            with torch.no_grad():
                logits = trainer.model(x_tr, y_tr, x_te)
                metrics[f"nll/val_dim{d}"] = trainer.bar_dist(logits, y_te).mean().item()
                metrics[f"mse/val_dim{d}"] = (trainer.bar_dist.mean(logits) - y_te).square().mean().item()
        return metrics

    return Callback(name="", fn=probe, every_n_steps=every_n_steps)


if __name__ == "__main__":
    from anytimeacquisition.callbacks.handler import CallbackHandler
    from anytimeacquisition.models.pfn import PFN

    class _FakeTrainer:
        def __init__(self, model):
            self.model = model
            self.bar_dist = model.bar_dist

    torch.manual_seed(0)
    max_x_dim = 4

    # The shared ecdf_sorted comes from a real BNNPrior -- here, one
    # spanning the same dimension range via variable_dim_min, mirroring
    # how train_pfn.py's main() sources it from the actual training prior.
    training_prior = BNNPrior(
        batch_size=8, x_dim=max_x_dim, variable_dim_min=1, seed=0,
        ecdf_n_draws=5, ecdf_samples_per_draw=50, cache_dir=None,
    )

    model = PFN(max_x_dim=max_x_dim, d_model=16, n_heads=2, n_layers=1, d_ff=32, n_bins=16)
    trainer = _FakeTrainer(model)

    callback = build_dim_validation_callback(
        dims=[1, 2, 4], max_x_dim=max_x_dim, ecdf_sorted=training_prior.ecdf_sorted,
        n_val_context=8, n_val_points=20, prior_kwargs=dict(depth_range=(8, 12), width_range=(16, 32)),
    )
    handler = CallbackHandler([callback])

    metrics = handler.run(step=0, trainer=trainer, default_every_n_steps=1)
    print("per-dimension validation metrics (untrained model, sanity-check only):")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")
    print("\nnote the metric-type-first keys (nll/val_dim1, mse/val_dim1, ...) -- "
          "all nll curves group together on a dashboard, all mse curves group together.")

    print("\nsame dedicated priors reused across two probes (architecture fixed, points resampled):")
    for step in (0, 1):
        m = handler.run(step=step, trainer=trainer, default_every_n_steps=1)
        print(f"  step {step}: nll/val_dim1={m['nll/val_dim1']:.4f}")
