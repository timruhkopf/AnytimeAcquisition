# ActionHead's explore-branch policy collapses to a context-independent uniform Beta, not overconfidence

**Date:** 2026-09-01
**Related:** `docs/ROADMAP.md` Phase 5 / M5, `src/anytimeacquisition/models/action_head.py`,
`src/anytimeacquisition/trainer/action_head_imitation_trainer.py`, `docs/log/2026-08-28-exploit-search-target-may-outrun-context.md`

## Motivation / hypothesis

The real-scale explore-only run (`useful-rat-611`, x_dim=6, `action_head_imitation_explore_real.yaml`)
showed `policy_nll/train` collapsing to near-zero (5.5 -> 0.008) while held-out
`l1/explore` barely moved (1.82 -> 1.70) and `auc/action_head` stayed *worse*
than `auc/random` throughout. Initial read (in conversation, not logged) was
"overconfident point-mass collapse" — this entry corrects that and pins down
the actual mechanism.

## What we tried

Reproduced the collapse at the cheapest possible scale to iterate fast:
`action_head_imitation_explore_smoke.yaml` (x_dim=1, `pfn_smoke_xdim1.pt`,
300 rollouts, otherwise default) run locally after landing this session's
DAgger phase-in fix (commit `1b8f179`, branch `explore-entropy-collapse`).
Then inspected `ActionHead`'s raw `alpha`/`beta` output directly (not just
the derived `policy/beta_entropy` metric) on a fresh held-out rollout.

`ActionHead.forward` (`models/action_head.py`): `alpha = softplus(x) + 1.0`,
`beta = softplus(x) + 1.0` — both floored at exactly 1.0, uncapped above.
`Beta(1,1)` is the uniform distribution on `[0,1]`; its density is exactly 1
everywhere, so `-log_prob(target) = 0` for **any** target in `(0,1)`. This is
a mathematical property of any proper NLL loss over a bounded, unit-measure
domain (`BarDistribution.entropy`'s own docstring/demo in
`models/bar_distribution.py` documents the same "uniform -> entropy exactly
0.0" fact for the PFN's own head), not specific to the Beta parameterization.

Control: ran the exploit branch the same way (`action_head_imitation_exploit_smoke.yaml`,
same PFN checkpoint) to check whether this is a loss-design pathology
(would show up regardless of target quality) or specific to explore's
noisier, single-restart (`n_restarts=1`) oracle targets. Exploit's targets
are tightly anchored near the incumbent (`exploit_search`'s own incumbent-
seeded, local-only redesign, `docs/log/2026-08-28-exploit-search-target-may-outrun-context.md`)
— much lower variance, so if exploit collapses too, that implicates the loss
itself rather than target noise.

## Result

Explore branch, 300 rollouts, held-out rollout inspection at steps 0/5/10/14,
8 batch instances each — `alpha`/`beta` both sit at **1.003-1.005** (i.e.
`softplus(x) ≈ 0.003-0.005`, essentially the exact floor) at every step and
every instance, barely distinguishable from each other:

```
step 0:  alpha=[1.0040, 1.0038, 1.0036, 1.0037, 1.0035, 1.0036, 1.0038, 1.0044]
         beta =[1.0045, 1.0042, 1.0047, 1.0044, 1.0048, 1.0045, 1.0045, 1.0037]
step 14: alpha=[1.0042, 1.0039, 1.0036, 1.0038, 1.0036, 1.0037, 1.0037, 1.0044]
         beta =[1.0035, 1.0038, 1.0043, 1.0041, 1.0044, 1.0042, 1.0042, 1.0035]
```

`policy/beta_entropy` sitting flat at ~0.00 for the entire 280+ remaining
rollouts (not diverging further negative) is consistent with this: 0.0 is
the uniform distribution's *own* entropy on `[0,1]`, the natural resting
point of this collapse, not a symptom of continued sharpening. `l1/explore`
staying flat around 0.27-0.32 the whole run also matches: `beta_mode`
returns exactly 0.5 in the `alpha=beta=1` degenerate case, and mean absolute
deviation of a Uniform(0,1) target from a constant 0.5 guess is 0.25 —
almost exactly what's observed. `blind_ratio/explore` hovering at ~1.0
throughout also fits: if the network isn't using context at all, real and
blind naturally produce nearly identical (both near-uniform) output.

Exploit-branch control (200 rollouts, same PFN checkpoint, same script
shape, in-process — `diag_exploit_collapse.py` below) tells a completely
different story: `policy_nll/train` keeps descending well below zero
(0.47 -> 0.02 -> **-0.02 -> -0.44 -> -0.72** by rollout 199) — density
genuinely exceeding 1 at the target, i.e. real, continuing peaking, not a
plateau. Direct `alpha`/`beta` readout confirms it's non-uniform and
non-trivial:

```
step  0: alpha mean/min/max = 1.21/1.00/2.39   beta mean/min/max = 2.25/1.00/5.39
step  7: alpha mean/min/max = 1.24/1.00/2.62   beta mean/min/max = 2.35/1.00/5.54
step 14: alpha mean/min/max = 1.05/1.00/1.28   beta mean/min/max = 2.41/1.00/5.43
```

`beta` consistently >> `alpha` (asymmetric, not the symmetric uniform
floor) — same architecture, same loss, same checkpoint, meaningfully
different training data, meaningfully different (successful) outcome. This
rules out "the Beta-NLL parameterization/ActionHead architecture is
inherently broken" as the primary explanation — it demonstrably learns
fine here.

**Follow-up: is explore's target actually predictable from context at
all?** `explore_search` (`search/explore.py`) seeds every restart at
`x_realized` — the point the rollout's *own* policy actually played at
that step (a deliberate 2026-09-01 correction, replacing an earlier fresh-
random-draw scheme, specifically so the search corrects what was played
rather than searching independently — see that module's own docstring).
Under `random_policy` (round 0, and still most of a `dagger_decay_rounds`
run since beta decays slowly), `x_realized` is literally `torch.rand(...)`
— pure noise, **statistically independent of context**. Measured directly
on 1283 explore-labeled examples across 20 fresh `random_policy` rollouts
(x_dim=1):

```python
# corr(x_star, x_realized)  = 0.7659
# corr(x_star, incumbent_x) = -0.2134
# var(x_star) = 0.1167;  var(x_star - x_realized) = 0.0490
# mean |x_star - x_realized| = 0.1714
```

`x_star` correlates strongly (r=0.77, ~59% of variance) with `x_realized`
— a quantity the ActionHead, which only ever sees `context`, cannot
observe — and only weakly (r=-0.21) with the incumbent, the one
context-visible quantity most analogous to what exploit's target is
anchored to. Exploit's target (a local refinement of the incumbent, which
*is* context-visible) is therefore fundamentally more learnable than
explore's (dominated by a quantity outside the policy's input) — not
because of anything about the loss, but because of what the two oracles'
targets are seeded from.

## What we learned

**This isn't (primarily) a loss-design bug — it's the explore branch's
oracle target being partly generated from information the policy can't
see.** The Beta-NLL floor (`softplus(x)+1`, giving `Beta(1,1)` = uniform =
*exactly* zero loss for any target) is real and is what lets the collapse
land precisely on the uniform distribution rather than some other
degenerate point — but the control run proves this same floor doesn't
cause collapse when the target is actually learnable from context. The
dominant cause is that `explore_search`'s `x_realized`-seeding (itself a
deliberate, differently-motivated fix from earlier the same day) means a
large fraction of `x_star`'s variance is, from the ActionHead's point of
view, irreducible noise — and predicting close to the (near-uniform,
since `x_realized` is uniform under `random_policy`) marginal distribution
of an unpredictable target is close to the actual NLL-optimal response,
not a training pathology to "fix" away by regularizing the loss alone.

Secondary, unresolved question: does this heal itself as DAgger's
self-play phase-in (this session's other fix, commit `1b8f179`) replaces
`random_policy` with the ActionHead's own (context-dependent, if
imperfect) actions over a run? The 300-rollout smoke run above already
reached `dagger/beta=0.05` (97% self-generated) by its end without
resolving — `l1/explore`'s history (`0.32, 0.28, 0.32, 0.30, 0.27, 0.32`
across all 6 logged AUC-eval ticks) is flat/noisy the entire run, not
trending down as self-play took over. Plausible reading: once the network
has settled into the near-zero-loss uniform basin (by rollout ~10, while
beta was still ~0.97), the gradient signal to climb back out is weak
regardless of how much better later self-play-derived data gets — this
wasn't directly tested (would need e.g. a longer run, or a fast-vs-slow
`dagger_decay_rounds` comparison) and is flagged as the natural next
experiment, not concluded here.

## Status / next steps

**Root cause identified, not yet fixed.** Candidate directions (not
mutually exclusive, none implemented yet — needs a decision, not more
diagnosis):

1. Downweight or delay explore-example collection until `dagger/beta` has
   decayed past some threshold (self-play meaningfully dominant), instead
   of training on it from rollout 0 -- directly cuts the volume of
   `x_realized`-dominated noisy targets the network ever has to reconcile
   against later, better data.
2. Re-open `explore_search`'s seeding choice specifically for early/
   high-beta rounds (e.g. blend `x_realized` with a context-derived point)
   -- in tension with the 2026-08-28/2026-09-01 design history that
   deliberately moved *toward* `x_realized`-seeding for a different,
   still-valid reason (genuine correction of what was played); revisiting
   it needs to preserve that reasoning for the self-play-dominant regime,
   not just revert it wholesale.
3. Test whether a much longer run, or a faster `dagger_decay_rounds`
   (self-play dominant much earlier), actually escapes the basin once
   reached -- the flat `l1/explore` trend above suggests "just train
   longer" alone may not be enough, but this wasn't directly isolated.
4. Curriculum: warm-start the ActionHead's shared blocks on the (cleanly
   learnable) exploit branch before ever training on explore targets.

Reported back to the user with this log for the next-step decision.
