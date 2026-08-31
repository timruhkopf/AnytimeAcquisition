# Variable-x_dim PFN training stagnated with per-instance active_dim — reverted to batch-uniform (ifBO-style)

**Date:** 2026-08-31
**Related:** `docs/milestones/M2.md`, `docs/OPEN_QUESTIONS.md` #2 (medium-dim
target), `src/anytimeacquisition/priors/bnn.py`'s `variable_dim_min`,
`src/anytimeacquisition/models/pfn.py`'s `n_features`/`_pad_and_rescale_features`

## Motivation / hypothesis

Earlier the same day, `BNNPrior`'s `variable_dim_min` option (already
present but unused) was wired end-to-end through `PFNTrainer` as the PFN's
`n_features`, to train a single checkpoint across a distribution of
`x_dim` values rather than one fixed dimensionality per checkpoint. That
first implementation sampled `active_dim` **per instance** — every one of
the `B` batch elements in a given `reset()` could land on a different real
dimensionality, all mixed into the same training batch/gradient step. This
was a deliberate choice at the time (see `models/pfn.py`'s and
`priors/bnn.py`'s docstrings as of that commit): it seemed strictly more
information-dense per step than ifBO/PFNs4BO's own convention, and this
repo's `BNNPrior` already vectorizes per-instance architecture diversity
(each batch element is its own randomly-drawn depth/width/weights), so
per-instance dimensionality diversity looked like a natural extension of
an already-established pattern rather than a new risk.

## What we tried

Trained a PFN checkpoint (MLflow run `popular-pig`, `x_dim=5`,
`variable_dim_min=1`, per-instance `active_dim`) via
`experiment=pfn_variable_xdim_smoke`-style config. The user reports it
stagnated and, on average across dimensions, underperformed
dimension-fixed ("marginal") checkpoints trained the ordinary way.

**Note:** this specific run lives in MLflow on `ulysses` or `LUIS`, not in
this machine's local `mlruns/` — it was not directly inspected as part of
this entry (checked all four local experiments, not present). Everything
below is reasoning from the user's report of the symptom, not from reading
the run's own loss curves directly.

## Result

Two candidate explanations were considered for "worse than marginal
models, on average over dimensions":

1. **Expected, not a bug**: with `n_train` capped at 100 (`max_train` in
   `PFNTrainer`) and `x_dim` up to 5, higher-dimensional instances get a
   much sparser context — the same 100 points cover a vanishingly smaller
   fraction of a 5D unit cube than a 1D one — so the PPD should genuinely
   be worse-calibrated there, pulling the batch-averaged NLL up relative
   to a model dedicated to one (easier, lower) dimension. This alone would
   explain "worse on average," but not stagnation.
2. **A real training-dynamics problem, specific to the per-instance
   design**: mixing differently-scaled instances (different
   `active_dim`, hence different `_pad_and_rescale_features` rescale
   factors and different effective signal density) into the *same* batch
   and gradient step is a meaningfully different — and untested —
   regime from ifBO/PFNs4BO's own proven convention, where the **whole
   batch** shares one `num_features` per step (their architecture/style
   diversity is likewise batch-level, not per-instance — see
   `docs/log/2026-08-27-pfns4bo-bnn-prior-comparison.md` point 2). Only
   the harder-in-high-dim signal in (1) has precedent; the added noise
   from within-batch dimension-mixing does not.

Acted on (2): `BNNPrior.reset()` now draws **one** `active_dim` shared by
all `B` instances (still resampled fresh every `reset()`, i.e. every
training step — exactly how `n_train` is already resampled per step in
`PFNTrainer`), instead of one `torch.randint(..., (B,), ...)` per
instance. This is a small, surgical change (one line in `reset()`); no
change was needed in `models/pfn.py` (its `n_features` mechanism already
handles a batch-uniform tensor as the default/simplest case) or in
`PFNTrainer` (it already just forwards `prior.active_dim` verbatim).
Updated the affected docstrings/comments in `bnn.py`, `pfn.py`,
`pfn_trainer.py`, and the two relevant config files accordingly, and
replaced the one test (`test_variable_dim_active_dim_in_range`) that
explicitly asserted per-instance variation with two tests checking the
new invariants (batch-uniform within one `reset()`; varies across
repeated `reset()` calls). Full suite green (91 passed) after the change.

## What we learned

**We do not know for sure that (2) was the actual cause of `popular-pig`'s
stagnation** — no controlled rerun (same seed/config, per-instance vs.
batch-uniform, holding everything else fixed) has been done yet. What we
do know: the per-instance design was untested against any working
reference, batch-uniform sampling matches a convention that ifBO/PFNs4BO
have already shown works, and it's a strictly simpler regime to optimize
(one dimensionality's worth of rescaling per step, not up to `batch_size`
different ones at once). That combination made reverting the reasonable
default move even without a confirmed root cause — but if a rerun at
batch-uniform still stagnates or still underperforms marginal models on
average, explanation (1) (or something else entirely — capacity dilution
from sharing one `d_model`/`n_layers` budget across a distribution of
dimensionalities, insufficient steps, warmup/LR tuned for a fixed-dim
regime, etc.) becomes the more likely story, and per-instance mixing can
be ruled out.

## Status / next steps

**Adopted** (batch-uniform `active_dim` is now the only mode
`variable_dim_min` supports) — reverting to per-instance would mean
undoing the `reset()` change and the test/docstring updates above, all in
one place. Not yet re-verified with an actual rerun of
`pfn_variable_xdim_smoke` (or an equivalent config) to see whether
stagnation persists at batch-uniform sampling — that's the natural next
step before drawing any further conclusion. A per-dimension NLL/eval_mse
breakdown (via the `Callback`/`CallbackHandler` mechanism in
`callbacks/handler.py`) was discussed as a way to actually see whether
higher dimensions are pulling the average up as explanation (1) predicts,
independent of whatever caused the stagnation — not implemented yet,
parked pending the rerun above.
