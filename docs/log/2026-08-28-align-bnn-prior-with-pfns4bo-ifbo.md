# Align BNNPrior with PFNs4BO/ifBO: noise, input scaling, sparseness, spurious dims, deeper default

**Date:** 2026-08-28
**Related:** `docs/milestones/M1.md`,
`docs/log/2026-08-27-pfns4bo-bnn-prior-comparison.md` and its addendum,
`docs/log/2026-08-27-bnn-prior-flat-draws-crit-scaling.md`

## Motivation / hypothesis

Yesterday's comparison against PFNs4BO/ifBO's own BNN priors surfaced
several mechanisms we didn't have: preactivation/output noise, fan-in-aware
input scaling, weight sparseness, and spurious (irrelevant) input
dimensions. User also wanted a deeper `depth_range` — PFNs4BO/ifBO never
sample below depth 8, ours went down to 2, which is structurally close to
incapable of complex output regardless of `crit`.

## What we tried

Implemented in `BNNPrior`:
- `depth_range` default raised `(2,17) -> (8,32)`.
- `sparseness=0.145`: each `W_h` (hidden-to-hidden only, matching
  `linears[1:-1]` in both reference implementations) weight zeroed with
  this probability, survivors rescaled by `1/sqrt(1-sparseness)`.
- `preactivation_noise_std_range=(0.0003,0.0014)` and
  `output_noise_std_range=(0.0004,0.0013)`, sampled per instance, injected
  at every layer (including the first) and at the output respectively.
  Added `noise: bool` to `_raw_forward`/`evaluate` so callers can get a
  deterministic surface at the current weights without changing the family
  config — needed later for M5's exploit/explore search, which wants an
  exact surface to differentiate through, not a fresh noise draw per call.
- Fan-in-aware input scaling: `x_scaled = (x - 0.5) / sqrt(1/12) / sqrt(d)`
  before the first layer, matching PFNs4BO's `sample_input()` +
  `x_ / sqrt(num_features)`.
- Spurious dimensions: each input dim independently "relevant" with
  probability `frac_relevant_features=0.7` (30% irrelevant, matching
  PFNs4BO §5.2); irrelevant dims get their `W_in` row zeroed per instance,
  so they're literally not fed to the network, not just given a dummy
  value.

## Result

All existing tests pass unchanged. New tests added
(`tests/test_bnn_prior.py`): noise toggle determinism (`noise=False` gives
bit-identical repeated calls, `noise=True` doesn't), sparseness actually
zeros a real fraction of `W_h` (vs. ~0 at `sparseness=0`, with a tolerance
for the vanishingly rare exact-zero float draw), spurious dims get exactly
zero gradient while relevant dims don't (checked directly, not inferred).

**Depth increase alone caused a regression, caught by looking at the
plots, not by the numbers.** Raising `depth_range`'s ceiling to 32 while
leaving `crit_range=(2.0,8.0)` fixed (the previous entry's fix) produced
visibly pure-noise environments at high depth — confirmed in
`plot_2d_environments()`'s output, one panel was indistinguishable from
static. This matches what the depth×crit sweep in the previous entry
predicted (`depth=16, crit=16` was already "jagged"; deeper still with the
same `crit_range` is worse) but hadn't been checked against the *new*,
wider depth range until now.

**Fix, confirmed with the user before implementing (not silently
changed)**: replaced `crit_range` with `log_amp_range=(8.0,20.0)`, a target
range for the *compounded* quantity `depth * log(crit)` rather than `crit`
alone. Per instance: sample `log_amp` uniformly from this range, derive
`crit = exp(log_amp / depth)`, then `init_std = sqrt(crit/width)` as
before. Re-ran the same 1D/2D visual check after the fix — no more static,
all sampled environments (depths spanning 11–31 in the check) show
coherent, richly-structured surfaces.

## What we learned

Same lesson as the previous entry, one level up: **a fixed target for a
quantity that multiplicatively compounds with something else that varies
(here, depth) doesn't stay fixed in effect** — `crit_range=(2,8)` was tuned
implicitly against the old `depth_range=(2,17)`'s effective range, and
silently stopped being valid the moment `depth_range` changed, with no
error or warning, just visually-obvious garbage output. Whenever a family
parameter changes, re-check whatever *other* parameter was implicitly
tuned against its old range — don't assume independence between
"unrelated-looking" hyperparameters without checking. Also: the plots
caught this immediately; the loss/metric numbers wouldn't have (a "flat"
region and "pure noise" region can both look unremarkable in an aggregate
statistic) — reinforces why the `__main__`-demo convention matters, not
just for interacting with a component but for catching this class of bug
at all.

## Status / next steps

Adopted. `log_amp_range` replaces `crit_range` in `BNNPrior` and
`configs/priors/bnn.yaml`. Input warping remains explicitly shelved (see
the 2026-08-27 addendum) — not implemented, revisit at real-world testing.
`docs/milestones/M1.md` updated to track what's now implemented.
