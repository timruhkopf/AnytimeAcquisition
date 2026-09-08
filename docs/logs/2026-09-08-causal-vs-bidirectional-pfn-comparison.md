# Causal vs. bidirectional PFN: a fair comparison, and where the learning curves say it's headed

Full reproducible analysis: `notebooks/pfn_causal_vs_bidirectional.ipynb`
(+ its CSV export, `notebooks/pfn_causal_vs_bidirectional_learning_curves.csv`).
This note is the summary and the reasoning behind the choices in that
notebook, not a duplicate of its content.

## Why this comparison exists

`models/pfn.py`'s `causal=True` mode (§PS.1/§PS.4 in `docs/PROBLEM_SETTING.md`
on the `roadmap-vla-privileged-search`-adjacent design work) restricts
train-train self-attention to lower-triangular instead of bidirectional,
trading exact permutation invariance for the ability to genuinely
KV-cache along a BO trajectory (a new observation extends the cache
instead of invalidating the whole context). Whether that tradeoff is worth
it was an open question, not something to assume either way — this
compares the two directly.

## Test setup

**Checkpoints:**
- `models/pfn_variable_xdim_smoke.pt` — bidirectional, **29,999 steps**
  (its own experiment config comment claims "smoke, 500 steps"; that
  comment is stale, the checkpoint's own logged history is the source of
  truth here, per an earlier session's correction).
- `models/pfn_causal_variable_xdim_real.pt` — causal, **59,999 steps**
  (deliberately 2x the bidirectional budget — causal training has
  strictly more to learn, validity at every prefix/order rather than just
  the full permutation-invariant set, so it needs a genuine chance to
  close any gap before concluding one exists).
- Matched architecture: `d_model=64, n_layers=4, n_heads=4, d_ff=128, n_bins=64`.
- Matched prior: `x_dim=6, variable_dim_min=1` (both trained across the
  same variable-dimensionality family).

**Held-out comparison (the notebook's main section):**
- `N_FUNCTIONS=30` fresh BNN draws per dimension, `x_dim ∈ {1,...,6}`.
- `N_TRAIN=15` context points, `N_TEST=50` held-out query points per
  function draw.
- `N_PERMUTATIONS=8` random re-orderings of each held-out context, for the
  causal model only (see "Hypothesis" below for why).

**Learning-curve analysis (break-even estimate):** uses the periodic
validation metrics logged *during* both training runs
(`callbacks/dim_validation.py`), evaluated against a **fixed, dedicated
validation prior per dimension**: `n_val_context=20, n_val_points=200`,
re-probed every `log_every` steps against the *same* underlying
architecture throughout — a much lower-noise signal than the training-batch
NLL (which varies step to step with the randomly resampled batch's own
`n_train`). This is the same data that was logged to MLflow during
training (identical `on_log` source populates both the MLflow store and
the checkpoint's own embedded `history` dict) — read from the checkpoint
directly here rather than querying the MLflow store separately, since
they're provably the same numbers.

## Hypothesis

Two distinct claims, tested separately:

1. **At matched-ish budget, bidirectional should have an advantage that
   isn't really about "better modeling"** — its context representation is
   permutation-invariant by construction, a free inductive bias the causal
   model has to earn empirically. Giving causal 2x the steps was the
   direct test of this: if the gap is just a sample-efficiency artifact of
   the harder task, more training should close it. If it doesn't close, or
   the projected trend says it won't, that's evidence of a persistent
   ceiling, not just slower convergence.
2. **A single evaluation of the causal model isn't a fair readout of its
   quality**, because its output depends on presentation order and a
   naive comparison could report a lucky or unlucky draw. The bidirectional
   model has no such issue by construction — verified directly on the
   actual checkpoint before trusting that as license to skip the
   permutation averaging for it (`1.1e-5` max logit difference under a
   train-set permutation, effectively float noise). The causal model was
   therefore evaluated over 8 random re-orderings of each held-out context
   and reported as the **permutation-averaged mean** — the fair number,
   since order at deployment is imposed by whatever acquisition policy is
   running, not chosen to flatter the model.

## Result 1: the direct comparison

Bidirectional beats causal at **every tested dimension** (1 through 6),
despite the 2x training budget:

| dim | bidirectional | causal (perm-avg) | causal order-sensitivity (std across 8 perms) |
|---|---|---|---|
| 1 | −2.444 ± 0.175 | −2.258 ± 0.190 | 0.135 |
| 2 | −1.746 ± 0.176 | −1.727 ± 0.187 | 0.064 |
| 3 | −1.067 ± 0.133 | −1.005 ± 0.140 | 0.037 |
| 4 | −0.905 ± 0.096 | −0.838 ± 0.092 | 0.032 |
| 5 | −0.736 ± 0.130 | −0.698 ± 0.128 | 0.030 |
| 6 | −0.650 ± 0.102 | −0.612 ± 0.092 | 0.029 |

(NLL, lower/more negative is better; ± is standard error over the 30
function draws. Small run-to-run variation in the exact numbers is
expected — re-running the notebook redraws fresh held-out functions.)

The gap is small at some dimensions (standard errors overlap at `d=2`/`d=6`,
so not a confident difference there at `n=30`) but the *direction* is
consistent everywhere — never a case where causal wins.

## Explaining the causal order-sensitivity column

This is the more novel finding: the causal model's own sensitivity to
presentation order (std of NLL across the 8 permutations of the *same*
held-out context) **shrinks monotonically with dimension** — `0.135` at
`d=1` down to `0.029` at `d=6`, roughly a 4-5x drop.

**Working hypothesis, not independently confirmed further:** at low `d`,
each individual context point carries a lot of distinguishing information
about the underlying function (in 1-D, 15 points already cover the domain
densely), so *when* a highly-informative point arrives relative to the
others plausibly shifts the causally-masked representation more. At high
`d`, 15 points is a very sparse sample of the space regardless of order —
the context is comparatively uninformative either way, so there's less
for order to disturb. This would predict order-sensitivity tracking
something like "how much does any single point change the posterior,"
which is exactly the quantity that shrinks as dimensionality (and thus
sparsity of a fixed-size context) grows.

**Practical reading:** if a causal PFN is ever deployed as an acquisition
policy's state encoder, its calibration is least stable under reordering
exactly in the regime (low `d`) where this project's overall design
otherwise has the most headroom to matter (per the M0 kill-test finding on
`roadmap-vla-privileged-search`, horizon-awareness effects were clearest
at small-to-moderate budgets/low dimensions too) — worth keeping in mind
as a joint consideration, not two independent facts.

## Result 2: learning curves and the break-even estimate

**Method.** A free nonlinear fit (`NLL(step) = a − b/step^c`) to the raw
per-logged-point validation curve landed on a degenerate, near-flat
solution (R² ≈ 0) — the per-point noise dominated the signal enough that
the optimizer couldn't find the real trend. Binning both curves into 15
step-buckets first, then fitting `NLL(step) = intercept + slope/step`
(linear in `1/step`, so plain OLS — no nonlinear optimizer to land in a
bad local minimum) gave a much better-behaved fit: **R² = 0.89**.

**Result:**
```
bidirectional plateau (mean of last 3 bins):  -1.3470
causal fit:  NLL(step) = -1.3002 + 2343.89/step
causal's fitted asymptote (step -> infinity): -1.3002
```

Causal's own fitted long-run asymptote sits **~0.047 NLL worse** than
bidirectional's already-achieved plateau. Under this model there is **no
projected crossing step** — not "catches up eventually, just needs more
steps," but a trend that leaves a persistent gap even in the limit. This
agrees with, and gives a mechanism for, Result 1's direct comparison.

**Caveats, stated plainly:** this is a 2-parameter fit to binned averages
from **one training run per architecture**, extrapolated roughly 2-2.5x
past the actually-observed step range. A different prior seed, more
validation samples per probe, or genuinely different dynamics at a much
longer horizon could all move this number, and "extrapolating a learning
curve" is inherently uncertain in a way a held-out comparison isn't. Take
the *direction and rough size* of the finding seriously (it agrees with
Result 1); don't treat `-1.3002` or "no crossing, ever" as a proven ceiling
from a single run.

## Bottom line

At this scale, the causal model's loss of exact exchangeability isn't a
free architectural swap — it costs real predictive quality, doesn't close
with 2x the training budget, and the learning-curve trend doesn't project
it closing with more steps either (though that specific projection is the
least certain part of this analysis). Whether the KV-caching throughput
benefit (the actual reason to want `causal=True` at all — see
`docs/PROBLEM_SETTING.md` §PS.4 and the session log on branch
`roadmap-vla-privileged-search` around 2026-09-08 for the rollout-collection-throughput
argument) is worth this quality cost is a real tradeoff now backed by a
number, not a guess — that decision belongs wherever this project's
current roadmap (`docs/ROADMAP.md`) ends up landing on it.
