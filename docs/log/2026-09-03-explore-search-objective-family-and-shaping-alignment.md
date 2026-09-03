# Explore-branch weighted-NLL objective: which acquisition family it's in, and whether it's safe as a dense shaping signal for the sparse exploit target

**Date:** 2026-09-03
**Related:** `search/explore.py`, `search/exploit.py`, `docs/milestones/M5.md`,
`pipelines/explore_search_playground.py`, `metrics/inc_auc.py`,
`notebooks/explore_search_labeling.ipynb`

## Motivation / hypothesis

Design conversation (not a code change) prompted by a direct question:
exploit-branch labels (`search/exploit.py`, GD from the current incumbent on
the true surface) are a *sparse* ground-truth improvement signal — they only
exist at rollout steps where the incumbent actually changed, and training
toward them directly drives the incumbent-AUC (`metrics/inc_auc.py`) down
whenever they fire. Most rollout steps aren't incumbent-improving, so the
explore branch's weighted-NLL objective (`search/explore.py`,
`improvement_weights`) exists to give the policy *something* to train on
densely, everywhere else.

The question raised: when the ActionHead is trained on these dense explore
labels, is it learning a genuinely more strategic exploration behavior (the
stated goal — minimize regret as fast as possible), or is it just learning a
slow, distilled reconstruction of an already-known myopic heuristic like EI?
And separately: is training on this dense proxy signal actually *safe* with
respect to the sparse, ground-truth exploit signal, or can it pull the
learned policy toward something that trades away long-run regret for
short-run NLL reduction?

## What we discussed

**1. Objective family.** EI/PI/UCB are each a function of *one candidate's
own marginal posterior* — `EI(x) = E_{y~posterior(x)}[max(0, incumbent −
y)]` never looks at the model's belief anywhere except at `x`.
`explore_search`'s objective is structurally different: it measures how
conditioning on a candidate `x` changes the model's NLL at a whole *set* of
other points `x_int`, weighted by `improvement_weights`. That's an
information/value-of-information criterion, not an improvement criterion —
mathematically in the Entropy Search / Predictive Entropy Search / Max-value
Entropy Search family, not the EI/PI/UCB family. One caveat worth keeping
attached to this: real entropy-search methods define their "where might the
optimum be" candidate set and weights from the *model's own* posterior;
`explore_search` uses privileged ground truth (`y_int_true`) for both,
because it's generating offline imitation-learning labels rather than making
a live acquisition decision — so it's closer to "an oracle-computable stand-in
for what an entropy-search objective would want" than to entropy search
itself.

**2. The checkable signature of "beyond EI."** The one thing a multi-point
aggregate objective can do that no single-candidate heuristic can even
express: pick a probe that disambiguates two competing candidate basins,
useful precisely *because* of its effect on other points, even when it isn't
individually the top-EI or top-UCB point (already noted in `search/explore
.py`'s own module docstring as the reason weighting alone doesn't collapse
onto the exploit branch's target). This gives a concrete, empirical test —
not just a philosophical one: correlate the trained policy's own chosen
probes (or `x_star` itself) against the argmax of EI/UCB/posterior-variance,
reusing `action_head_ei_diagnostic.py`'s correlation machinery (built for a
different target, trivially adaptable). High correlation → the policy
converged on an amortized version of something already known — a legitimate,
useful, but not novel result. Low correlation *combined with* competitive or
better regret/AUC → real evidence of behavior a myopic heuristic can't
represent.

**3. Reward-shaping framing: the potential-based case, and what breaks it.**
Sparse exploit labels are the true target; explore labels are a denser,
correlated proxy added to speed up learning where the true signal is silent
— exactly the reward-shaping problem in RL. Ng, Harada & Russell (1999)
identify the one specific form of shaping term that's provably safe: given
any potential function `Φ(s)` over states alone, define

```
F(s, a, s') = γ·Φ(s') − Φ(s)
```

Summed over any complete trajectory `s0 → s1 → ... → sT`, this telescopes:

```
Σ_t F(s_t, a_t, s_{t+1}) = γ^T·Φ(s_T) − Φ(s_0)
```

— a term depending only on where the trajectory started and ended, not on
which actions were taken along the way. Because of that, adding `F` to the
true reward `R` shifts every policy's expected return by the same kind of
state-dependent offset and can never flip the ranking between two policies
(or two actions) that `R` alone would give: the optimal policy under `R+F`
is exactly the optimal policy under `R`. Shaping that is *not* of this exact
telescoping form has no such guarantee — it can change what's optimal, not
just how fast you find it.

The one finding worth keeping from this line of reasoning: **if
explore-branch and exploit-branch training signals actually compete** — the
gradient pulling toward the explore label at a given state sometimes points
away from what the exploit label would say if one existed there — **then
this is, by construction, not the telescoping/safe case.** A telescoping
bonus can't create a per-step conflict with the true objective, because it
never depends on the action taken at all, only on the states visited; a
genuine tug-of-war between two per-step targets is exactly the signature of
a shaping term that isn't potential-based. So "do the two branches compete"
isn't just an empirical curiosity — it's the direct, checkable stand-in for
"is this shaping actually safe," precisely because the one case the
theorem's guarantee covers (a state-only potential difference) structurally
cannot produce that competition in the first place.

## What we learned

- The objective genuinely sits in a different, and appropriate, family
  (information-directed, not improvement-based) for the stated goal — this
  is a structural fact, not just a hope, and it's specific to *why*
  `explore_search` sums weighted NLL over a whole `x_int` set rather than
  scoring `x` alone.
- That structural fact does not by itself certify either (a) that the
  *trained policy* ends up behaviorally novel rather than reconstructing a
  known heuristic, or (b) that training on this dense signal never trades
  away regret at the (sparse, ground-truth-correct) exploit-labeled states.
  Both are open, and both are empirical, not analytic, questions given the
  tools currently in the repo.
- `pipelines/explore_search_playground.py`'s `greedy_regret` diagnostic is
  the only existing tool that checks "does weighted-NLL improvement actually
  track regret reduction" — and it has only ever been run against the hard
  `improvement_weights`, on the *search's* output (`x_star`), never against
  a *trained policy's* behavior, and never against the soft-weighting
  variant discussed alongside this in the same session.

## Follow-up: the actual spectrum for driving down log-inc-AUC, and two refinements

**Motivation.** Direct follow-up: since `noise=False` makes the environment
deterministic, fully known, and differentiable, and we have privileged
access to it for label generation, is single-step behavioral cloning really
the best use of that privilege, or is there a genuinely more direct way to
minimize log-inc-AUC?

**The optimal-control framing.** A deterministic transition (`f(x)`) plus a
deterministic, known cost (log-inc-AUC as a function of the observed `y`
sequence) is a finite-horizon deterministic optimal control problem.
Bellman's principle applies: a value function `V_t(context)` — "the best
achievable AUC contribution from here to the end" — exists, and the truly
optimal policy is `π*(context) = argmin_x [cost(x) + V_{t+1}(context ∪
{x,f(x)})]`. Neither `exploit_search` (1-step, direct-improvement-only) nor
`explore_search` (1-step, weighted-NLL-only) is this `V` — both are myopic
approximations to it.

**The spectrum, in order of lookahead:**
1. Current M5 — 1-step myopic search (`exploit_search`/`explore_search`),
   then BC. Cheap, stable, biased toward locally-correct/globally-wrong
   choices.
2. k-step receding-horizon privileged planning (Guided Policy Search
   style): jointly GD-optimize a short sequence `(x_1..x_k)` against a
   k-step-ahead objective, teacher-forced through the true BNN, keep only
   `x_1*` as the label, re-plan next step. Directly generalizes machinery
   already built (`exploit_search`/`explore_search`'s own multistart GD).
3. Full-horizon BPTT — differentiate the entire episode's (softmin-relaxed)
   AUC straight into the ActionHead's parameters `θ`. Mathematically the
   most direct route to the exact objective, but inherits and compounds the
   multi-basin local-optima problem `exploit_search`/`explore_search`
   already need multistart restarts for at a SINGLE step, plus standard
   BPTT vanishing/exploding-gradient and memory costs over a full rollout.
   Agreed in this session to not be practically viable here.
4. The exact Bellman-optimal solution via a value function — intractable
   exactly (curse of dimensionality over possible contexts), approximated
   via a learned value head. This is already `docs/ROADMAP.md` Phase 5.5's
   plan (`A_t = G_t − V_φ(s_t)`,
   `docs/log/2026-09-01-explore-fallback-and-credit-assignment-open-questions.md`).

**Refinement 1 — is (2) actually feasible, given a PFN forward pass is
needed at every step?** Raised directly: doesn't jointly optimizing
`(x_1..x_k)` require re-running the PFN at every one of the `k` steps
per gradient iteration, reintroducing the same per-step compute blowup that
makes (3) infeasible, just with a smaller multiplier?

The answer depends on how the k-step objective is scored, and there's a
cheap version: score the PFN only ONCE per gradient iteration, against the
context obtained by adding all `k` candidate points at once — not once per
intermediate step. That single evaluation already reflects what a
`k`-point-ahead context would tell the model (weighted NLL at `x_int`,
exactly `explore_search`'s existing objective, just conditioned on `k` new
points instead of 1) — this costs about the same as today's single-step
`explore_search`, plus a slightly longer context per PFN call, not `k`
separate calls. A "sum over intermediate states" version would cost more
(up to `k` PFN calls per gradient step), but even that remains
fundamentally more tractable than (3), for a structural reason, not just a
smaller constant: `x_1..x_k` here are `k` independent FREE TENSORS being
directly gradient-descended (exactly like `candidates` in
`exploit_search`/`explore_search` already), scored through a FROZEN PFN
used as a static feature extractor at each evaluation. There is no shared,
LEARNABLE network being recursively applied and backpropagated through
time — that recursive-shared-weights structure is specifically what makes
(3) hard (`θ` must receive a correctly-propagated gradient through every one
of `T` sequential applications of itself), and (2) never has it, regardless
of how many PFN calls the scoring uses. `k` is also a small, fixed planning
horizon, decoupled from the total episode length `T` (which can be several
times larger) — so even the more expensive scoring variant does not scale
with episode length the way (3) does.

**Refinement 2 — `γ`-discounting vs. budget-conditioning for the planned
value function.** Raised directly: a discount `γ` in `V`/the advantage
estimate imposes a fixed, exponentially-decaying temporal preference: it
doesn't know anything about the episode's actual structure, only "how many
steps away." But this project's `remaining_budget` is a known, causally-
visible quantity already present as a canonical `ActionHead` aux feature
(`canonical_aux_features` in `pipelines/action_head_ei_diagnostic.py` and
`pipelines/action_head_posterior_distill.py`) — and "anytime" acquisition
is explicitly about behaving well relative to a finite, known-remaining
budget, not an infinite or unknown one. This is a real and correct
critique, not just a preference: standard finite-horizon MDP theory treats
this exactly as a *time-indexed* value function `V_t(s)` (equivalently, one
conditioned on remaining horizon), not a stationary `γ`-discounted
approximation — `γ`-discounting is itself normally motivated as a
relaxation for *infinite or unknown*-horizon problems, which this problem
is not. The natural fix, when M5.5's value head is designed, is to
condition `V_φ` on `remaining_budget` directly (already available as an
input) rather than relying on a fixed `γ` to encode "how much do I still
care about the future" — the value function can then learn the correct,
possibly non-exponential shape of that preference from data. One nuance
worth not conflating with this: `γ` (or a TD-λ-style parameter) can still
have a legitimate, SEPARATE role purely as a variance-reduction device in
how a return/advantage is *estimated* from finite rollout samples (e.g.
GAE) — that's a bias/variance tradeoff in estimation, not a claim about
what the true objective's time preference should be, and budget-
conditioning `V_φ` doesn't automatically resolve it.

**Refinement 3 — the k-step joint score is not `x_1*`'s own value, and the
PFN's permutation invariance is exactly why.** Raised directly: the k-step
objective scores `context ∪ {x_1,...,x_k}` as one set through a single PFN
call; because the PFN reads context as a permutation-invariant set (no
notion of "x_1 arrived before x_2"), that score is a property of the whole
set, not of `x_1` individually. But only `x_1` is ever actually deployed —
`x_2,...,x_k` are discarded and replaced by real re-planning. So does the
score that justified picking `x_1*` say anything about what `x_1*` is
worth alone?

Two distinct claims need to be separated here, and only one survives:

- *Valid:* using the joint objective to **select** `x_1*`. This is the
  standard model-predictive-control argument — `x_1`'s job is to lead to a
  context from which *some* good continuation exists, not to deliver the
  whole k-step improvement by itself, and the two-hills-style intuition
  from the main discussion (a point that looks unremarkable alone can be
  correctly favored because of what it sets up) is exactly this. It holds
  precisely because real replanning happens at every actual step, from the
  real observed context — `x_2*..x_k*` never need to be real or correct
  for `x_1*`'s selection to have been sound.
- *Invalid:* treating the joint score `val_star` as **`x_1*`'s own
  delivered value**. A team's collective value is not automatically
  attributable to one member — this is precisely the problem Shapley
  values exist to solve in cooperative game theory. `x_1` deployed alone
  reaches the smaller context `{x_1}`, not `{x_1,...,x_k}`, and its real,
  standalone NLL improvement at `x_int` is generally smaller than the
  k-step score, possibly by a lot if most of that value actually came from
  `x_2` or `x_3`. Because context is read as an unordered set, there is no
  way to decompose the joint number back into per-point shares after the
  fact — the attribution is genuinely gone, not just hard to compute.

The fix is the discipline this session's notebook already established for
a different reason (never trust `explore_search`'s own reported `val_star`
as ground truth — recompute independently, "the second forward pass"):
after the joint k-step search picks `x_1*`, always run one more, honest,
single-point forward pass — `weighted_nll_after(context ∪ {x_1*})`, alone,
nothing else added — and treat *that* number, not the joint score, as
`x_1*`'s real contribution. Worth being explicit that the permutation
invariance itself is correct behavior, not a flaw: a static, non-time-
varying function's posterior genuinely shouldn't depend on the order two
observations arrived in. The mistake would only be in conflating "the
model correctly ignores order when reading a *completed* context" with
"the k planned points were interchangeable *while being chosen*" — they
weren't; only `x_1` is real, the rest are leverage for finding it, and
their apparent value cannot be assumed to transfer to it.

## Status / next steps

Open — no mechanism or verification procedure decided, recorded as a
problem statement. Concrete follow-ups flagged, none yet started:

1. Re-run an `explore_search_playground.py`-style regret-correlation check
   with `soft_improvement_weights` swapped in for `improvement_weights`,
   to see whether the correlation between weighted-NLL improvement and
   `greedy_regret` reduction holds up in the "residual signal, no real gap
   exists" regime the soft weighting specifically targets.
2. Build a classical-acquisition-argmax correlation diagnostic for the
   *trained ActionHead's own outputs* (not just the search's), reusing
   `action_head_ei_diagnostic.py`'s machinery, to empirically test whether
   the trained policy's behavior resembles EI/UCB/PI or diverges from them
   while maintaining competitive regret — the concrete test for "did it
   learn something beyond the classical family."
3. An ablation, since no formal potential-based-shaping-style guarantee
   exists here: train one ActionHead on exploit labels only, another on
   exploit + explore, and compare their behavior *specifically at
   exploit-labeled decision points* (held fixed across both runs). If
   adding the dense explore signal degrades performance there, that's
   direct evidence of the harmful-interference risk described above, not
   just a theoretical possibility.
4. Eventually, the actual milestone-level test: head-to-head incumbent-AUC/
   regret comparison of the fully-trained policy against EI/UCB/PI/random
   baselines on held-out environments. Everything above is about whether
   the training signal is well-posed enough to make that comparison worth
   running, not a substitute for running it.
5. Prototype a k-step privileged planner with FINAL-STATE-ONLY PFN scoring
   (one PFN call per gradient iteration, against the context obtained by
   adding all `k` candidate points at once) — the concrete, tractable
   version of spectrum level (2) above, and a direct generalization of
   `exploit_search`/`explore_search`'s existing multistart-GD machinery.
   Must apply Refinement 3's discipline from the start: report/compare
   `x_1*` only via an independent single-point re-scoring, never via the
   joint plan's own objective value — a first prototype now exists,
   `notebooks/kstep_explore_search_labeling.ipynb`, which measures the gap
   between the two directly.
6. When M5.5's value head is designed, condition `V_φ` on `remaining_budget`
   (already an existing `ActionHead` aux feature) instead of, or alongside,
   a stationary `γ` discount — per finite-horizon MDP theory — keeping `γ`/
   `λ` only if needed for return-estimation variance reduction, not as the
   source of temporal preference.
