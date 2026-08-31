# Explore-branch search: from a self-referential entropy objective to teacher-forced, privileged-NLL gradient descent

**Date:** 2026-08-28
**Related:** `docs/milestones/M5.md`, `docs/ROADMAP.md` Phase 5, `search/explore.py`,
`search/interesting_points.py`, `trainer/exit_rollout.py`
(`build_explore_buffer`), `pipelines/explore_search_playground.py`

## Motivation / hypothesis

M5's explore branch needs an oracle target for rollout steps where the
incumbent didn't improve (`trainer.exit_rollout.label_branches`'s
complement). Design direction (explicit user call, this session): rather
than the design doc's original "posterior-plausible points from the
model's own current posterior," build a **fixed, per-episode** set of
"interesting" test points (`x_int`) from Sobol + random + GD-restart-found
basins on the BNN's own true surface (`search/interesting_points.py`),
weight them by privileged log-improvement over the current incumbent
(`weight_i = max(0, log(incumbent) - log(y_int_true_i))`,
`improvement_weights` — the same log-improvement quantity
`metrics/inc_auc.py`'s `log_incumbent_stepwise_reward` uses on a realized
trajectory), and find a candidate query `x_explore` via gradient descent
that reduces (weighted) uncertainty at those points. Hypothesis: summing
across a *weighted set* rather than optimizing toward a single best point
would avoid the explore branch collapsing onto the same target the exploit
branch already produces ("exploitation-pull").

## What we tried

**v1 — entropy objective, self-predicted y (the flawed version).** Optimize
`x_explore` to minimize `sum_i weight_i * entropy_i`, where `entropy_i` is
the frozen PFN's closed-form entropy at `x_int_i` given `x_explore` added
to context. Since `x_explore`'s true y isn't known in advance, v1 imputed
it from the frozen PFN's *own* posterior mean at `x_explore` under the
current context (a first forward pass), then fed that guess back into the
same PFN as the "observed" value for a second forward pass to score entropy
at `x_int`.

Wired into `trainer/exit_rollout.py`'s `build_explore_buffer` (mirroring
`build_exploit_buffer`, but for explore-labeled steps) and run end-to-end
against `pfn_smoke_xdim1.pt` on a real rollout: **15/19** corrections
reduced weighted entropy vs. doing nothing.

**The circularity, caught by the user directly:** "we don't want the pfn's
gradient control what the y value is from its perspective the 'self
serving' y." v1's first pass asks the frozen PFN "what do you think you'd
see here?" and then treats that guess as ground truth for the second pass.
The optimizer can reduce *reported* entropy by finding `x_explore` where
the model's own prediction happens to be self-consistent with what it
already believes — not by finding locations that carry real information
about the true function. An under-calibrated model can exploit that gap
outright.

**v2 — teacher-forced y, still entropy.** Ground `x_explore`'s y in the
BNN instance's own true, noise-free surface instead:
`y_true = prior.evaluate(x_explore, noise=False)`, recomputed **fresh at
every GD step** from whichever `x_explore` value that step holds — not
cached at initialization, not a free/learnable variable of its own. `prior`
is "pre-hooked" as a frozen-but-differentiable module ahead of the PFN:
its own parameters are never learnable, `x_explore` is the *only* free
tensor in the whole graph, and gradients reach it through two real paths —
directly via the location term in the PFN's train-token embedding, and via
`d(y_true)/d(x_explore)` through the BNN's own Jacobian. This is not "run
GD against a fake y, then patch the answer afterward" (post-hoc
overwrite) — the gradient trajectory itself is grounded in reality at
every step, because a genuinely different (correct) `y` is computed at
every step from wherever `x_explore` currently is. Same privileged-access
justification `search.exploit.exploit_search` already relies on (training-
data-generation only, never deployment) — no new privilege boundary
introduced. One PFN forward pass cheaper per step, too: no separate
self-prediction pass needed once the true y is available directly.

Re-ran the identical rollout/seed: **19/19** corrections reduced weighted
entropy. The jump from 15/19 confirmed the circularity was a real,
measurable failure source, not just a theoretical concern.

**Remaining risk, named but not fixed by teacher forcing:** gradient-based
input optimization against a frozen, imperfectly-calibrated network is
structurally similar to adversarial-example generation. Teacher forcing
closes off the "invent a self-serving y" exploit specifically, but the
*second* pass (entropy/NLL at `x_int` given the augmented context) still
runs through the same frozen, possibly-undercalibrated PFN, and `x_explore`
is still being aggressively searched against that network's output. A
concrete plausible failure mode: `x_explore` converging near an
already-observed context point — a near-duplicate `(x,y)` configuration
PFN pretraining rarely produces (`sample_episode` draws distinct uniform-
random points), i.e. exactly the kind of out-of-training-distribution
input a network's behavior is least trustworthy on. The `[0,1]` clamp
bounds the *domain*, not the *distribution* — this risk is open, not
resolved.

**Revisiting an earlier overclaim.** The "summing across a weighted set
avoids exploit-collapse" argument is real but not absolute: as the weight
distribution over `x_int` concentrates (e.g. one BNN instance has one
sharp global optimum well ahead of everything else — entirely plausible
given `log_amp_range`), nearly all the weight mass sits on one location,
and minimizing weighted entropy/NLL degrades toward "reduce uncertainty at
that one point" — i.e. converges toward what the exploit branch would
target anyway. `find_basins` keeping every restart's endpoint (not just the
best) spreads *coverage* of `x_int`, but doesn't flatten the *weight*
distribution, since weight is a function of each point's true y value, not
of how many nearby candidates exist. Whether this is actually a problem is
arguable (a genuinely needle-in-a-haystack instance arguably *should* pull
explore toward the one region that matters) — but the earlier framing
("avoids collapse" without qualification) was stated too confidently.

**Entropy → NLL (user-directed, "entropy can be confidently wrong").**
Entropy measures the PFN's own confidence, blind to whether that confidence
is *correct* — a model can reduce it by becoming falsely sure of the wrong
answer. Since `y_int_true` is already privileged/known for every point in
`x_int`, a strictly harder-to-game objective was sitting right there:

```
minimize  sum_i weight_i * NLL(y_int_true_i | PPD at x_int_i given x_explore added)
```

using `bar_dist.forward` (the bar-distribution's *native training loss*,
already implemented/tested) in place of `bar_dist.entropy`. NLL against a
*known* true value can't be reduced by unearned confidence — confidently
wrong makes it sharply worse, not better — while entropy would reward it.
It's also asymmetric in a useful way for a fixed-bin bar distribution:
bounded reward for being confidently right (`-log(n_bins)`, can't go much
lower), unbounded penalty for being confidently wrong. Framed against the
Bayesian-optimization literature, this whole mechanism (fixed candidate
set, privileged weighting by decision-relevance, gradient search for an
informative next query) sits closest to **Knowledge Gradient** and
**loss-calibrated / decision-focused active learning** — value of
information judged by how much it would improve a specific downstream
decision, not generic uncertainty reduction.

**A more literal alternative, considered and parked (user: "don't like the
argmin thing").** The most decision-theoretically direct version of
"reduce regret" would simulate the model's own greedy choice after the
hypothetical update: `argmin_i predicted_mean_i`, scored against the true
`y_int_true` at that index. `argmin` isn't differentiable; a
softmin-over-predicted-means relaxation would fix that, but adds a new
temperature hyperparameter and an unvalidated mechanism, with no evidence
NLL is insufficient. **Not built as the optimization objective.** The same
quantity turned out to be exactly the right tool for a different job,
below — as a non-differentiable, post-hoc *validation* metric it needs no
relaxation at all, since nothing there differentiates through it.

**Implementation + regret validation.** Swapped `_weighted_entropy` for
`_weighted_nll` in `search/explore.py` (`explore_search`'s signature,
mechanism, and best-so-far tracking are otherwise unchanged). Re-verified:
`search/explore.py`'s own demo (3 fresh instances) — 3/3 reduced weighted
NLL, one instance showing a large swing (2.32 → -0.21), consistent with
NLL's sharper penalty/reward structure. `trainer/exit_rollout.py`'s
rollout demo — 19/19 again, now measured in NLL rather than entropy.

Built `pipelines/explore_search_playground.py` specifically to answer the
question NLL-improving alone doesn't: does this correction reduce true
regret, not just the proxy? `greedy_regret(context) =
y_int_true[argmin_i predicted_mean_i] - y_int_true.min()` — the true
regret of the model's own greedy pick among `x_int`, computed before and
after adding each explore correction, across many real rollouts.

## Result

10 episodes, 314 explore corrections, against `pfn_smoke_xdim1.pt`:

| metric | value |
|---|---|
| mean regret before | 0.0631 |
| mean regret after | 0.0613 |
| mean regret reduction | 0.0017 |
| fraction with regret strictly improved | 18.5% |
| fraction with regret strictly worsened | 14.3% |
| mean weighted-NLL improvement | 0.645 |
| **correlation(NLL improvement, regret reduction)** | **0.863** |

## What we learned

- **Post-hoc y-overwriting and per-step teacher forcing are not the same
  mechanism, and the difference is load-bearing.** If the gradient
  trajectory itself never touches the true value, no amount of correcting
  the final answer afterward makes the *search* trustworthy — the
  optimizer only ever chases what it was actually shown at each step.
- **Teacher forcing measurably fixed a real problem (15/19 → 19/19), but
  did not make this search fully trustworthy.** It closes the
  self-referential-guess exploit specifically; the broader
  "gradient-search-against-a-frozen-network" risk (adversarial/out-of-
  distribution inputs) is a distinct, still-open concern that scales with
  how well-calibrated the checkpoint is, not something teacher forcing
  touches.
- **NLL against a privileged known value dominates entropy for this
  specific setup** (training-time search, true value available) — not just
  "an alternative," a strictly harder-to-game choice given what's already
  known, at zero extra engineering cost (reuses the model's own training
  loss).
- **Argmin/softmin-based literal regret is the right tool for validation
  and the wrong tool for the optimization objective**, and those are easy
  to conflate. Needing a metric non-differentiable is a reason to keep it
  out of the loss, not a reason to discard the metric — `greedy_regret`
  ended up being exactly what closed the loop empirically.
- **The 0.86 correlation is real evidence the proxy works directionally,
  but the low improved/worsened split (18.5% vs. 14.3%) is an honest,
  unflattering number worth keeping, not smoothing over** — most
  corrections don't flip the model's greedy pick at all; NLL improving is
  necessary-looking but not sufficient for "this specific correction
  mattered." Both numbers came from a smoke-scale checkpoint
  (`pfn_smoke_xdim1.pt`) — re-measure before trusting either figure against
  a converged model.
- **The exploit-collapse-avoidance claim needed walking back to a
  qualified version.** It holds when `x_int`'s weight mass is genuinely
  spread; it degrades toward exploit-like behavior as weight concentrates
  on one dominant optimum. This interacts with the still-open multi-basin
  exploitation-pull fix (M5.md) from a different angle than originally
  scoped — worth re-checking together, not separately, when that fix is
  built.

## Status / next steps

**Adopted:** teacher-forced y (BNN pre-hooked, frozen-but-differentiable,
only `x_explore` learnable) + weighted NLL is the current
`search.explore.explore_search` implementation.
`pipelines/explore_search_playground.py` exists specifically to re-run the
regret-validation numbers above whenever a non-smoke checkpoint becomes
available — do that before trusting explore-branch outputs in a real EXIT
training run, since both headline numbers here (0.86 correlation, 18.5%/
14.3% split) are checkpoint-quality-dependent, not architecture-dependent.

**Parked, not rejected:** the softmin-relaxed literal-regret objective as
a *replacement* for NLL in the optimization loop. Revisit only if the
NLL-based version's regret-correlation turns out insufficient on a better
checkpoint — there's no evidence yet that it is.

**Still open, unaffected by this fix:** the adversarial/out-of-distribution
input-optimization risk (teacher forcing doesn't touch it), and the
multi-basin exploitation-pull fix for the exploit branch (M5.md) — the two
are related through the weight-concentration mechanism named above.
