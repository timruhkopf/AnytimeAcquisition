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

## Fix: seed at the incumbent instead of `x_realized`

Chosen over the other three candidates below (commit `bd9d1de`, branch
`explore-entropy-collapse`): `explore_search`'s restart seed changed from
`x_realized` to the incumbent (`x_context[argmin(y_context)]`) — the same
context-visible anchor `exploit_search` already uses, which the control
run above proved trains cleanly. Keeps the "stay local, don't roam blind
on a tiny context" property `x_realized`-seeding had (a different
context-anchored point, not a reversion to the earlier "fresh independent
random draw" design already rejected once, see
`docs/log/2026-08-28-explore-search-input-optimization-and-teacher-forcing.md`).
Renamed the parameter `x_realized` -> `x_seed` throughout
(`search/explore.py`, `trainer/exit_rollout.py::build_explore_buffer`,
`callbacks/action_head_validation.py::build_explore_signal_rate_callback`).
Known open trade-off: loses "correct what the policy actually proposed"
for later, self-play-dominant rounds — deferred, see "Open follow-ups"
below.

**Acceptance criterion** (added to `docs/milestones/M5.md`): held-out
`auc_improvement_vs_random/mean > 0` (the project's own north-star metric,
`callbacks.action_head_validation.build_auc_eval_callback`), sustained
over the last 3 logged eval ticks.

**Result at x_dim=1** (`action_head_imitation_explore_smoke.yaml`, 300
rollouts, otherwise unchanged from the pre-fix run above — direct
before/after comparison):

```
                              before fix          after fix
policy_nll/train (final)      +0.005              -0.172   (was NEVER negative in 300 rollouts;
                                                              after fix, negative from rollout 10 on)
policy/beta_entropy trend     flat at ~0.00        diverges: -0.08 -> -0.46 (final)
l1/explore (last 6 ticks)     .32/.28/.32/.30/.27/.32   .20/.27/.22/.20/.17/.21
blind_ratio/explore (last 6)  1.05/.95/.92/.96/.95/.94  1.12/1.28/1.00/.89/.82/.89
auc/action_head (final)       -12.06 (worse than random)   -19.39 (better than random)
auc_improvement_vs_random     -3.33...-6.40 (always neg.)  -4.97,-0.77,1.56,2.39,1.62,0.93
  (all 6 ticks)                                            (last 3: all positive)
```

**Acceptance criterion MET at x_dim=1**: last 3 ticks of
`auc_improvement_vs_random/mean` are `[2.39, 1.62, 0.93]`, all positive
— action_head beats random search, sustained, not a single lucky tick.
Direct `alpha`/`beta` inspection was skipped this round (the AUC/entropy/
blind_ratio trends already tell a consistent, unambiguous story) but would
be a natural sanity re-check if this result needs re-litigating later.

**Real-scale (x_dim=6) confirmation**: launched on `ulysses` directly via
SSH (not through PyCharm's interpreter, to avoid the `/tmp/pycharm_project_*`
staging confusion — see `scripts/mlflow_tunnel_ulysses.sh`'s own
documented gotcha) — `action_head_imitation_explore_real.yaml` against
`pfn_variable_xdim_smoke.pt`, branch `explore-entropy-collapse` @ `bd9d1de`,
300 rollouts, checkpoint at `models/explore_seedfix_xdim6.pt`. **Still
collapsed**: `policy/beta_entropy` flat at ~0.00 from tick 2 on (same
signature as the pre-fix run), `l1/explore` flat ~1.68-1.75,
`auc_improvement_vs_random/mean` strongly negative every tick (-37 to
-52). The seeding fix alone did not transfer to real scale.

## Retraction: the x_dim=1 "criterion met" claim didn't survive scrutiny

User pushback (correctly): the training run's own 6 AUC-eval ticks all
reuse the SAME fixed 8 held-out instances (`eval_seed=999`) — "3
consecutive positive ticks" isn't 3 independent trials, and 1D is exactly
where random search is hardest to beat anyway. Re-evaluated the trained
x_dim=1 checkpoint properly:

```
same eval_seed=999, n=8  (sanity check): improvement mean=0.70  95% CI=[-2.46, 3.87]  -- includes 0
DIFFERENT eval_seed=2027, n=40 (real test): improvement mean=-1.82 95% CI=[-4.23, 0.58] -- negative, includes 0
  fraction of instances action_head beat random: 0.375 (worse than a coin flip)
```

**Correction: the acceptance criterion was NOT met at x_dim=1 either.**
The seeding fix's genuine, unambiguous achievement is fixing the *training
dynamics* (negative NLL, diverging entropy, blind_ratio < 1 — all real,
reproduced below) — it did not, on its own, produce a policy that
robustly beats random search.

## Reach diagnostic at x_dim=6: the seeding fix's target IS genuinely informative

Before assuming "search doesn't reach far enough from the incumbent in
6-D" explains the x_dim=6 non-transfer, measured it directly (10 fresh
`random_policy` rollouts, real config's own `n_restarts=1, n_steps=15`):

```
n examples: 6064
mean ||x_star - x_seed|| (L2, 6 dims): 0.6181   (vs. max possible ~2.449)
mean |x_star - x_seed| per-dim: 0.2025
var(x_star) per-dim ≈ var(x_seed) per-dim ≈ 0.089;  var(delta) per-dim: 0.070
corr(x_star, x_seed) per dim: 0.52-0.66   (vs. 0.77 for the broken x_realized case)
```

The search moves substantially and produces a target only moderately
correlated with its seed — genuinely context-informative, not "incumbent
plus noise." Ruled out: this isn't a search-reach problem. The remaining
collapse points at the loss's own optimization dynamics, not target
quality — see the floor fix below.

## Fix #2: raise the alpha/beta floor from 1.0 to 2.0

Commit `653002e`. At floor=1.0, `Beta(1,1)` is *exactly* uniform — density
exactly 1 everywhere on `[0,1]`, so `-log_prob(target)=0` for any target,
a literal zero-cost point reachable regardless of how many dimensions the
policy_head has to jointly solve. At floor=2.0, `Beta(2,2)` is not
uniform — retreating to the floor no longer eliminates loss. Exploit's
own already-learned `alpha`/`beta` (observed up to ~5.4) sit comfortably
above 2.0, so exploit shouldn't regress (not separately re-verified this
round — flagged as a follow-up).

Re-ran the x_dim=1 comparison (300 rollouts, otherwise identical):

```
                          seeding-fix only     + floor=2.0
policy_nll/train (final)  -0.172               -0.841   (both negative; floor version more so)
policy/beta_entropy       -0.08 -> -0.46        -0.23 -> -0.14 (noisier, but never returns to ~0.00)
l1/explore (6 ticks)      .20/.27/.22/.20/.17/.21    .21/.29/.10/.10/.08/.09  (clearly, consistently better)
blind_ratio/explore       1.12/1.28/1.00/.89/.82/.89 1.05/.97/.84/1.10/.64/.77 (lower, more consistently <1)
auc_improvement_vs_random -4.97,-0.77,1.56,2.39,1.62,0.93 (all 6 ticks)  -6.43,-5.36,-5.36,-1.79,-3.43,-3.11
                          (looked positive, later shown not significant)  (consistently negative, smaller magnitude
                                                                            than pre-any-fix, but still negative)
```

The floor fix's per-rollout training loss was also visibly *more volatile*
(large swings, e.g. 3.18, 2.15, 2.46, 1.97 interspersed with negative
dips) rather than either cleanly collapsed or cleanly converging —
consistent with a network now actually grappling with fitting inconsistent
per-step targets instead of retreating to the free uniform answer.

**The decisive read**: `l1/explore` and `blind_ratio/explore` — both
direct measures of "does the network accurately reproduce the oracle's
own `x_star`" — improved clearly and consistently with the floor fix. But
`auc_improvement_vs_random` stayed negative regardless. This separates two
previously-conflated questions: *can the network learn the oracle's
target* (yes, now — both fixes together resolve this) vs. *is the
oracle's target itself good enough to produce a policy that beats random
search* (no, unresolved by either fix).

## Where the real gap is: a design-doc-documented step was never built

Per the user's suggestion, re-read `archive/src/exit/PFN_ActionHead_ExpertIteration_Design.md`
(§4, step 3, "Branch valuation") — the original Expert Iteration design
this repo is built from. It does **not** directly imitate a raw oracle
search result. Every candidate correction is scored with "a short, cheap
rollout... plus the current value head's bootstrapped estimate for
whatever trajectory remains" (the AlphaZero-style move) *before* it's
added to the training buffer. This repo's `build_exploit_buffer`/
`build_explore_buffer` skip that step entirely — whatever
`exploit_search`/`explore_search` returns is treated directly as ground
truth, unfiltered. `train_value_head` (the thing that would make
valuation possible) defaults to `False` and is unset in every existing
config.

This isn't a new concern invented today — `docs/milestones/M5.md`'s own
earlier regret-validation finding already measured the consequence: only
**18.5%** of individual explore corrections strictly reduced regret
(14.3% made it worse). Both fixes landed today make the network *learn
that signal faithfully* (proven: `l1/explore` down, `blind_ratio` < 1) —
neither fix changes the fact that the signal being learned is right about
1 in 5 times, unfiltered. That is a plausible, well-evidenced explanation
for why a demonstrably-non-collapsed, accurately-fitting policy still
doesn't beat random search.

## What we learned

Three real, distinct issues, all confirmed empirically, not just
theorized:
1. DAgger's rollout-mixing schedule was inverted (fixed, unrelated to
   this specific investigation — see commit `1b8f179`).
2. `explore_search` seeded at a quantity independent of context under
   `random_policy` (`x_realized`), making its target partly unlearnable
   in principle — fixed by seeding at the incumbent instead (`bd9d1de`).
   Confirmed real (training dynamics genuinely changed) but **not
   sufficient on its own** to produce a policy that beats random search,
   and did not transfer to x_dim=6 at all.
3. The Beta-NLL loss's `alpha=beta=1` floor gave a literal zero-cost
   "give up" point regardless of target quality, independent of (2) --
   raising it to 2.0 (`653002e`) makes the network demonstrably learn the
   oracle's target faithfully. Still not sufficient: `l1`/`blind_ratio`
   improve, `auc_improvement_vs_random` does not.

The likely dominant remaining gap is (per the reference design) the
missing branch-valuation/value-head-bootstrap step — without it, the
network is being trained to faithfully reproduce a training signal that's
only right about 1 in 5 times. Fixing the *learnability* of a noisy
signal doesn't fix the noise.

## Status / next steps

**Root cause chain understood, two of three contributing issues fixed
and confirmed real, acceptance criterion still not met at either scale.**
Not continuing to unilaterally build the missing branch-valuation
mechanism — it's a substantial feature (value-head training, a rollout-
based validation/filtering step in `build_exploit_buffer`/
`build_explore_buffer`, config wiring), not a bug fix, and needs a
decision, not more solo iteration. Reported back to the user with this
log for that decision.
