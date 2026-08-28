# PFNs4BO's own BNN prior — what they do differently from ours

**Date:** 2026-08-27
**Related:** `docs/milestones/M1.md`, `docs/milestones/M2.md`, `docs/OPEN_QUESTIONS.md` #2,
`docs/log/2026-08-27-bnn-prior-flat-draws-crit-scaling.md`

## Motivation / hypothesis

PFNs4BO (Müller, Feurer, Hollmann, Hutter, ICML 2023;
[automl/PFNs4BO](https://github.com/automl/PFNs4BO)) is one of the design
doc's Appendix references. Before continuing to build on our own BNN prior,
wanted to check what their actual reference implementation does — both to
sanity-check our design choices and to catch anything we should deliberately
adopt or deliberately diverge from, rather than reinvent blind.

## What we tried

Read `pfns4bo/priors/simple_mlp.py` (the BNN/MLP prior itself),
`pfns4bo/priors/hyperparameter_sampling.py` (how they randomize
architecture across a batch), `pfns4bo/bar_distribution.py` and
`pfns4bo/priors/normalize_with_style.py` (output-scale calibration), and
`Tutorial_Training_for_BO.ipynb`'s `config_bnn` (their actual training
config, with exact hyperparameter ranges) via the GitHub API (`gh api`).

## Result

**Confirmed provenance, bug included.** Our archived `bnn.py`'s numeric
constants — depth range, width range `36-150`, `init_std` range
`0.08896049884896237, 0.1928554813280186`, sparseness
`0.1449806273312999` — match `config_bnn`'s `ConfigSpace` hyperparameters
to the exact decimal. They were copied from this tutorial notebook, not
independently derived. And their `mlp_init_std` / `mlp_num_hidden` are
independent `ConfigSpace` hyperparameters too — no tying via anything like
our `crit` formula (see the flat-draws log entry). So the flatness bug we
found and fixed very plausibly also affects some fraction of their own
published training data.

**Structural differences, roughly in order of relevance to us:**

1. **No ECDF at all.** `simple_mlp.py` returns the *raw*, unbounded MLP
   output as `y`. Calibration happens in the model instead:
   `FullSupportBarDistribution`'s bin borders are fit *once*, empirically,
   from a batch of sampled `target_y` (`get_bucket_limits`). We ECDF-
   normalize the environment's output to `[0,1]` before the PFN ever sees
   it — a different component owns the same underlying problem
   ("what scale does this prior live at"). M2 is planning to port their
   *full* (not bounded) bar distribution — pairing that with our
   already-bounded-to-`[0,1]` prior is a mismatch with how they actually
   use it. **Open decision for M2, not yet made.**
2. **Architecture diversity is batch-level via `ConfigSpace`, not
   instance-level and not vectorized.** `num_hyperparameter_samples_per_batch`
   (16 of 128 in `config_bnn`) distinct architecture configs, each shared by
   several sequences, built via a plain sequential Python loop — no
   padding/masking, no vectorization across differing architectures at all.
   Confirms our earlier vmap reasoning (M1 log): they didn't vectorize the
   varying-architecture dimension either, they avoided the problem by making
   architecture batch-level instead of instance-level. Our masked-einsum
   approach gives more diversity per batch and stays vectorized.
3. **Preactivation + output noise**, both tiny (~0.0003-0.0014), injected at
   every layer and at the output; they distinguish the noisy `y` the PFN
   observes from an optional noise-free target (`mlp_noisy_targets`). We're
   fully deterministic. Probably a calibration/regularization choice, not a
   flatness fix, but a real modeling decision we're not making.
4. **Input scaling accounts for `x_dim`; ours doesn't.** They z-score the
   uniform input and divide by `sqrt(num_features)` before the first layer
   — a fan-in correction for input dimensionality itself. We feed raw
   `[0,1]` x straight into `W_in`. Our `crit` fix targets hidden-to-hidden
   transitions (fan-in = width); the first layer's fan-in is `x_dim`, which
   we don't compensate for. Didn't matter at the `x_dim=1` we tested;
   will matter more as we move toward the medium-dim target.
5. **Input warping** — `config_bnn` chains an extra randomized nonlinear
   input-space distortion after the MLP. We have nothing equivalent.
6. **They train one PFN across a distribution of `x_dim`** (up to 18
   features, via `sample_num_feaetures_get_batch`), not one fixed
   dimensionality per checkpoint. Directly bears on
   `docs/OPEN_QUESTIONS.md` #2 (medium-dim target still unpinned) — a real
   alternative to picking one value.
7. **Sparseness** (~14.5% of hidden weights zeroed, rescaled) is in their
   config and in our *old* archived prototype, but got dropped when M1 was
   based on `vectorized_bnn.py` instead.

## What we learned

Their reference implementation isn't automatically "more correct" than
ours where it matters most (the flatness bug exists there too, as far as
we can tell without literally training their model and checking) — but it
makes several deliberate structural choices worth weighing on their own
merits rather than by authority: output-scale calibration lives in the bar
distribution's borders, not in the environment; architecture diversity is
achieved by batch-level `ConfigSpace` sampling instead of per-instance
vectorization; and several small noise/warping/dimensionality-generalization
choices exist that we don't currently make. None of these are obviously
"the right answer" for us — several depend on decisions we haven't made yet
(M2's bar distribution, the `x_dim` target) — but they're now informed
choices instead of unexamined defaults.

## Status / next steps

Not yet acted on. **Scheduled as the next work item starting 2026-08-28**:
resolve finding #1 (ECDF vs. adaptive bar-distribution borders) and #6
(fixed vs. variable `x_dim`) before proceeding further into M2, since both
are upstream of real M2 implementation choices. See the "Next up" note in
`docs/MILESTONES.md` — remove that note once this is picked up (or by
2026-08-29 regardless, since "next up as of 2026-08-28" stops being an
accurate description once that day has passed); this log entry stays as
the permanent record either way.

## Addendum (2026-08-28)

The user supplied the actual PFNs4BO paper text (Sections 4.3, 5.1, 5.2,
Appendix B.2), which sharpens two findings above.

**Finding #2 (batch-level architecture diversity), in the authors' own
words.** The paper states this plainly as a deliberate, acknowledged
trade-off, not an oversight:

> "To be more efficient, we sampled the hyperparameters (the standard
> deviation of the distributions and the architecture) 16 times per batch,
> but used a batch size of 128. This is more efficient than sampling these
> per example, but adds correlation to the gradients inside a batch."

Our masked-einsum approach doesn't face this trade-off at all — every
instance gets its own architecture *and* stays vectorized, no per-instance
Python loop. Worth being precise: this isn't just "we're more diverse,"
it's "we avoid a cost they explicitly named and accepted."

**Input warping — noted, deliberately shelved.** Appendix Section 5.1, in
the authors' words:

> "In addition to using input warping after prior-fitting (see Section
> 3.5), we can include a Bayesian formulation of input warping in the prior
> directly and include it in prior-fitting. That is, (i) we sample warping
> hyperparameters h_warp randomly from a predefined distribution and then
> (ii) warp inputs of a synthetic prior dataset with an exponential
> transform and hyperparameters h_warp. While we found this to be
> beneficial, it was not powerful enough to remove the need for feature
> warping after prior-fitting completely."

So it's used twice in their pipeline — once baked into the prior during
training, once again post-hoc on real inputs at deployment — and the
in-prior version alone wasn't sufficient on its own. We have neither.
**`SHELVED`** — not implemented now; revisit once we've reached real-world
testing (i.e. once the pipeline runs against actual, non-synthetic
hyperparameter search spaces, where warping's benefit — modeling
non-uniform input sensitivity — would actually be measurable). Tracked in
`docs/milestones/M1.md`.

Also newly implemented as a direct result of this comparison: preactivation
+ output noise, fan-in input scaling, sparseness, spurious dimensions, and
a raised `depth_range` — see
`docs/log/2026-08-28-align-bnn-prior-with-pfns4bo-ifbo.md` for that work,
including a depth/crit interaction it surfaced that this entry's `crit`
fix didn't anticipate.
