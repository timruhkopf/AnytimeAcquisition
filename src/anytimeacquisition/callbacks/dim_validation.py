"""Per-dimension validation callbacks (M2) -- built specifically to check
the hypothesis in `docs/log/2026-08-31-variable-xdim-training-stagnation.md`:
does a variable-x_dim PFN's NLL/eval_mse genuinely get worse at higher
dimensionality (curse-of-dimensionality, expected), independent of whatever
`BNNPrior.active_dim` happens to be doing on any given training step?

Each dimension in `dims` gets its own **dedicated, fixed-x_dim** `BNNPrior`
(`variable_dim_min=None`), built ONCE by `build_dim_validation_callbacks`
and reused for every subsequent probe -- not rebuilt per call. Two costs
this saves: the ECDF fit (a real, one-time cost per prior, see
`priors/bnn.py`'s own timing notes -- this is why they're built once at
setup, not inside the returned callbacks' `fn`) and, more subtly, task
drift -- `reset()` is called exactly once per dedicated prior (at
construction), never again, so every probe against a given dimension
samples fresh points from the *same* fixed underlying random
architectures throughout the whole training run, the way a held-out
validation set should behave -- not a fresh random task each time, which
would make the reported curve noisier and harder to read trends from.

Requires the model being validated to already accept the largest `dim` in
`dims` (i.e. `max(dims) <= trainer.model.max_x_dim`) -- checked eagerly at
build time, not discovered later at the first probe.
"""
from typing import Any

import torch

from anytimeacquisition.callbacks.handler import Callback
from anytimeacquisition.priors.bnn import BNNPrior

# Deliberately cheap by default -- this ECDF fit is purely for a fair,
# reasonably-calibrated eval-prior Y-normalization, not for matching the
# training prior's own precision (the model being validated already
# learned from whatever ECDF precision ITS OWN training prior used; these
# dedicated priors are separate instances regardless). See priors/bnn.py's
# constructor comment for what these three numbers cost.
_FAST_ECDF_KWARGS = dict(ecdf_n_draws=10, ecdf_samples_per_draw=100, ecdf_n_samples=200)


def build_dim_validation_callbacks(
    dims: list[int],
    max_x_dim: int,
    prior_kwargs: dict | None = None,
    n_val_context: int = 20,
    n_val_points: int = 200,
    seed: int = 0,
    every_n_steps: int | None = None,
    ecdf_kwargs: dict | None = None,
) -> list[Callback]:
    """One `Callback` per entry in `dims`, each named `val_dimN` and
    reporting `{"nll": ..., "eval_mse": ...}` against its own dedicated,
    fixed-x_dim `BNNPrior` -- so callers get e.g. `val_dim1/nll`,
    `val_dim5/eval_mse` as separate, directly comparable metrics (same
    dashboard-grouping convention as `trainer/pfn_trainer.py`'s own
    `train/nll`, `eval/mse`).

    `prior_kwargs` should mirror the training run's own `priors.*` config
    (`depth_range`, `width_range`, `log_amp_range`, `sparseness`, ...) so
    the comparison is apples-to-apples against the same task family --
    only `x_dim`/`variable_dim_min`/`batch_size` differ per dimension here
    and are set internally; passing any of those three raises, since
    silently overriding a caller's explicit value would be more confusing
    than refusing.  `every_n_steps=None` (default) uses the trainer's own
    `log_every` cadence (see `callbacks/handler.py`); building `len(dims)`
    ECDF fits already costs real wall-clock time up front, so this
    shouldn't also default to running more often than the trainer's own
    built-in eval.
    """
    reserved = {"x_dim", "variable_dim_min", "batch_size"}
    prior_kwargs = dict(prior_kwargs or {})
    clashing = reserved & prior_kwargs.keys()
    if clashing:
        raise ValueError(
            f"build_dim_validation_callbacks sets {sorted(reserved)} itself (per-dimension) -- "
            f"remove {sorted(clashing)} from prior_kwargs rather than overriding them here."
        )
    too_big = [d for d in dims if d > max_x_dim]
    if too_big:
        raise ValueError(f"dims {too_big} exceed max_x_dim={max_x_dim} -- the model can't accept them")

    ecdf_kwargs = dict(ecdf_kwargs if ecdf_kwargs is not None else _FAST_ECDF_KWARGS)

    callbacks = []
    for d in dims:
        val_prior = BNNPrior(
            batch_size=n_val_context, x_dim=d, variable_dim_min=None, seed=seed,
            **prior_kwargs, **ecdf_kwargs,
        )

        def probe(step: int, trainer: Any, val_prior: BNNPrior = val_prior) -> dict:
            x_tr, y_tr, x_te, y_te = val_prior.sample_episode(n_train=n_val_context, n_test=n_val_points)
            with torch.no_grad():
                logits = trainer.model(x_tr, y_tr, x_te)
                nll = trainer.bar_dist(logits, y_te).mean().item()
                eval_mse = (trainer.bar_dist.mean(logits) - y_te).square().mean().item()
            return {"nll": nll, "eval_mse": eval_mse}

        callbacks.append(Callback(name=f"val_dim{d}", fn=probe, every_n_steps=every_n_steps))
    return callbacks


if __name__ == "__main__":
    from anytimeacquisition.callbacks.handler import CallbackHandler
    from anytimeacquisition.models.pfn import PFN

    class _FakeTrainer:
        def __init__(self, model):
            self.model = model
            self.bar_dist = model.bar_dist

    torch.manual_seed(0)
    max_x_dim = 4
    model = PFN(max_x_dim=max_x_dim, d_model=16, n_heads=2, n_layers=1, d_ff=32, n_bins=16)
    trainer = _FakeTrainer(model)

    callbacks = build_dim_validation_callbacks(
        dims=[1, 2, 4], max_x_dim=max_x_dim, n_val_context=8, n_val_points=20,
        prior_kwargs=dict(depth_range=(8, 12), width_range=(16, 32)),
    )
    handler = CallbackHandler(callbacks)

    metrics = handler.run(step=0, trainer=trainer, default_every_n_steps=1)
    print("per-dimension validation metrics (untrained model, sanity-check only):")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")

    print("\nsame dedicated priors reused across two probes (architecture fixed, points resampled):")
    for step in (0, 1):
        m = handler.run(step=step, trainer=trainer, default_every_n_steps=1)
        print(f"  step {step}: val_dim1/nll={m['val_dim1/nll']:.4f}")
