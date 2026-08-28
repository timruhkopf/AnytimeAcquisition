# In-Context Anytime Acquisition: A Privileged-Search-to-RL Design

*Complete design document — reflects all corrections made through discussion,
including two structural gaps in the pure-imitation version (budget-blindness
and exploitation-pull) that motivate the RL extension.*

---

## 1. Problem Setting

Standard Bayesian optimization parametrizes the explore–exploit trade-off
with hand-crafted, ahead-of-time heuristics (EI, UCB, entropy search). A
Prior-Data Fitted Network (PFN) already gives a calibrated, in-context
Bayesian surrogate essentially for free. The goal here is to stop
hand-crafting the acquisition function on top of that surrogate and instead
**learn it end-to-end, directly against the objective actually cared about:
anytime performance over a full optimization trajectory**, operationalized
as log-incumbent AUC.

Two properties of this setting shape everything that follows:

- **Privileged information is available at training time only.** Because the
  synthetic prior is under our control, the true function, its optimum, and
  (via the frozen PFN's closed-form output) the exact predictive entropy
  anywhere are all known during training-data generation. This access
  disappears at deployment, where only real, costly evaluations exist.
- **The training objective (AUC) is trajectory-level; the cheap privileged
  signals are per-step.** This mismatch is the central tension of the whole
  design, developed in full in §6.

---

## 2. The BNN Prior

The synthetic data-generating process: each batch element is an
independently sampled random-architecture tanh MLP (random depth, random
width, random weight-init scale), giving one synthetic "true function" draw
per instance. Outputs are squashed to `[0, 1]` via a **differentiable ECDF**
fit once, at construction time, by Monte-Carlo sampling across the whole
architecture/weight *family* (not per-instance) — this is what makes the
marginal output distribution comparable across differently-shaped/scaled
sampled networks.

Everything in `evaluate(x)` is differentiable w.r.t. `x`. This is
intentional: it is the privileged, known, differentiable dynamics that the
exploit/explore privileged-search steps (§5) rely on.

Two properties discovered empirically, both worth carrying forward as known
behavior rather than surprises:

- **Achievable output range varies wildly by instance.** Some architecture
  draws saturate almost immediately and are nearly flat everywhere (observed
  ranges as narrow as 0.012); others span nearly the full `[0, 1]`. A search
  result that looks unimpressive can simply reflect an unusually flat
  instance, not a failing search — always check the instance's true range
  before judging a result.
- **Multi-fidelity is deliberately out of scope.** Extending to freeze-thaw
  multi-fidelity would require a mixed discrete/continuous action space (an
  increasing basket of already-evaluated configurations to choose among, vs.
  a continuous space for new ones) that materially complicates the acquisition
  problem. Shelved as an explicit phase-two extension, not an oversight.

---

## 3. The PFN Surrogate

Standard PFN attention pattern: train tokens self-attend to train tokens
only (bidirectional); test tokens cross-attend to train tokens only — never
to each other, never to themselves. This is what makes the train-token
representation a permutation-invariant summary of the context (matching what
a Bayesian posterior should be) and what prevents test-test leakage (so each
test point's prediction is conditionally independent given the context, as a
valid pointwise PPD requires). No positional encoding anywhere: train tokens
must stay permutation-invariant, and test tokens don't attend to each other,
so there is nothing for a position to disambiguate.

**Output head: bar (Riemann) distribution.** Logits over `n_bins` fixed
bins; each bin is a piece of constant density (density = softmax probability
÷ bin width), trained via proper continuous NLL. Gives a closed-form
differential entropy, `H = -Σ p_i log(p_i / width_i)`, needing no Monte
Carlo — this closed form is what makes the explore-branch search (§5) a
cheap gradient step rather than a sampling-based one.

The implementation uses the **bounded** variant (fixed finite borders over
`[0, 1]`, since the BNN prior's output is guaranteed in that range), not the
open-tailed full-support version a real, unbounded HPO target would need.
Documented as a deliberate simplification appropriate to this specific prior,
not implemented for the general case.

**Empirical status.** Trained on CPU, `x_dim=2`, 750 steps: train NLL −0.66 →
−2.40, eval MSE down to ≈0.005. Permutation invariance and absence of
test-test leakage both verified numerically (not just asserted), to
residuals near floating-point zero. One open finding, not yet resolved: on
wide-output-range instances, predictive entropy does not shrink monotonically
with context size, while MSE does — most likely an artifact of shallow
training (750 steps is a smoke-scale run), but it means the entropy signal
should currently be trusted more on easy/narrow-range instances than on hard,
wide-range ones, which matters directly for the explore branch's reliability.

---

## 4. VLA-style ActionHead: Architecture and Distributional Head Options

### 4.1 Architecture

Modeled on the π0/π0.5 pattern: the PFN plays the role of the frozen
vision-language backbone; a separately-weighted ActionHead plays the role of
the action expert, reading the backbone's internal representations via
cross-attention rather than being handed only its final output.

- Cross-attends into the frozen PFN's **train-section hidden states**,
  tapped at **multiple layers**, not just the last — the final layer is
  compressed toward the NLL objective specifically and may attenuate
  information not needed for PPD prediction but needed for exploration
  decisions.
- **No gradients flow from the ActionHead into the PFN.** This mirrors
  π0.5-KI's explicit gradient block from the action expert into the VLM
  backbone, done there (as here) to protect the frozen backbone's original
  competence — for us, the PFN's calibration. Verified numerically: PFN
  parameter gradients are exactly zero after a backward pass through the
  ActionHead.
- **Explicit auxiliary tokens** carry step count, budget remaining,
  incumbent value, and recent-improvement trend. The PFN's train section is
  permutation-invariant by design — a calibrated posterior shouldn't depend
  on arrival order — so none of this is guaranteed recoverable from the
  cache, and it is injected explicitly rather than assumed implicit.

**Empirical verification, not just structural claims:**

- *Memorization*: policy MSE 0.377 → 0.002 over 150 epochs on 8 fixed
  examples — confirms the cross-attention pathway is correctly wired and has
  real capacity.
- *Ablation vs. a "blind" ActionHead* (PFN hidden states zeroed, aux features
  untouched): the real version fits training data far better (policy MSE
  0.025 vs. 0.116) — the cross-attention pathway carries real signal, not
  decoration, since the blind version structurally cannot distinguish
  episodes that differ only in where context points landed.
- *Generalization across many random instances*: initially mixed — the value
  output generalized cleanly, the policy output overfit. Isolated by
  re-running with a single fixed instance (only the context subset varies):
  the overfitting vanished. **Conclusion: the original generalization gap was
  a data-scale/task-diversity artifact of a 48-example static dataset trying
  to generalize across a fresh random function every episode, not an
  architecture flaw.**

### 4.2 Distributional head: the mode-averaging question

A plain point-regression policy head risks **mode-averaging**: if training
targets for similar states are inconsistent — landing in different, both
locally-optimal regions — MSE regression blurs them into a bad compromise
between the modes rather than committing to either. Three candidate fixes
were weighed:

| Option | Verdict | Reasoning |
|---|---|---|
| Bar distribution factorized over `x`'s dimensions | **Rejected** | Loses cross-dimensional correlation, and joint (non-factorized) binning is combinatorially impractical as dimension grows. |
| Full flow matching (as in π0/π0.5) | **Rejected for now** | π0.5's justification is *genuine* multimodality in human demonstration data (different demonstrators, different valid solutions) combined with long action chunks and real-time inference constraints. Our targets come from a single deterministic oracle search per call, not multiple human demonstrators — the setting that earns flow matching's complexity doesn't fully transfer here. Disproportionate machinery for a single low-dimensional point target. |
| **Mixture Density Network (MDN)** head | **Adopted as the pragmatic middle ground** | Handles genuine multimodality where it exists, single forward pass at inference (no iterative denoising), far less machinery than flow matching. |

**Before building the MDN, a cheap, falsifiable check should be run first,
and has not yet been run**: is the multimodality real, or self-inflicted by
the oracle's own randomized restarts? Take fixed states from a trained run,
call `exploit_search` on each with several different random seeds, and
measure the spread of resulting `x*`. Tight clustering → the concern was
mostly search-noise, not real landscape multimodality. Genuinely separated
clusters → real multimodality (plausible, given BNN architectures up to
depth 16 can have multiple comparable local optima), and the MDN earns its
place. Two candidate fixes considered and explicitly rejected as insufficient
on their own: canonicalizing the search (always restart from the same
anchor) and reusing the bar-distribution machinery for `x` (rejected per the
table above) — the MDN remains the standing choice regardless of what the
diagnostic shows, since even self-inflicted inconsistency is a real training
signal an MDN would absorb more gracefully than point regression would.

If RL is reached (§7), the MDN stops being optional: policy-gradient updates
require differentiating `log π(x)`, which a point-regression head does not
provide. At that point the MDN is a structural requirement, not a
mode-averaging mitigation.

---

## 5. Expert-Iteration-style Imitation: Exploit/Explore Oracle Design

### 5.1 Branch mechanics

**Exploit branch.** A few steps of gradient descent directly on the **true**
prior surface (`prior.evaluate`, differentiable by construction), started
both from points on the current trajectory and from random restarts. Cheap —
one differentiable function, no PFN forward pass required.

**Explore branch (as corrected through discussion — not iterative posterior
sampling).**

1. Obtain a small, *fixed* set of "interesting" points once, from privileged
   knowledge of the true surface — via the same random-restart search used
   by the exploit branch, attaching the (approximate) global optimum.
   Interesting points are **weighted by inverse (log) performance**, so
   exploration is continuously pulled toward the incumbents/optimum in
   proportion to how good they are, rather than toward an arbitrary fixed
   set. This makes the explore target genuinely *state-dependent* — the more
   context accumulates, the more informed the weighting becomes about where
   near-optimal actually is.
2. Pass those fixed points through the frozen PFN **once** as test queries,
   to get their current predictive entropy (closed-form).
3. Optimize a **single** candidate explore point via a few gradient steps,
   minimizing entropy at the fixed interesting points if the candidate (with
   its true `y`, from the differentiable prior) were added to the context.
   Only the candidate point is iterated; the interesting points are never
   re-optimized. Cost profile matches the exploit branch — a few forward/backward
   passes on one point, not an iterative sampling loop.

**Branch assignment: hard partition by realized outcome, not a learned
rule.** Steps where the true incumbent improved are labeled exploit; flat
steps are labeled explore. This is a structural partition, decided *after*
the fact from the actual trajectory, which is the main defense against the
two branches' gradients collapsing onto the same target (a real risk if
"interesting points" had instead been tied directly to ground-truth optimum
location without the state-dependent weighting in step 1 above).

### 5.2 On the relationship to Policy Gradient Search (Anthony et al., 2019)

PGS extends AlphaZero-style Expert Iteration to domains without a discrete
tree, replacing tree search with online gradient-based search initialized
from the current policy's own proposed action. Our search already does
something structurally similar — `exploit_search` is warm-started from
recent points on the trajectory the current ActionHead actually produced,
not from scratch — so the apprentice already shapes *where* search happens.

This was examined as a possible gap (should search also be warm-started from
the ActionHead's literal proposed action, more tightly coupling search
quality to apprentice quality, as PGS does) and **deliberately not adopted**:
PGS's search is graded only by local refinement of the network's own belief,
since it has no ground truth to fall back on — meaning a bad policy can
anchor the search into a bad basin that a few local steps can't escape, and
"correction" ends up being a polished version of the same mistake. Our
search is graded against the **true function**, insulated from ever scoring
the apprentice against itself, so it retains PGS's efficiency benefit
(locally relevant, a natural curriculum) without inheriting its
degeneration risk. This is treated as settled, not an open gap.

---

## 6. Trajectory-Level Critique: Why Pure Imitation Cannot Fully Optimize AUC

This section is the central finding of the design process, arrived at
through two specific, concrete failure modes rather than a generic "RL is
better" argument.

### 6.1 Budget-blindness

Neither `exploit_search` nor `explore_search` takes remaining budget as an
input to the optimization. On two states that are geometrically identical
but differ only in steps remaining, the oracle produces the *identical*
target. A budget-aware optimal policy should not — it should shift toward
exploitation as budget depletes. This is not a matter of the proxy being
imperfect and possibly underperforming; it is **structurally, provably**
blind to a variable already known to matter, and imitating the oracle
exactly reproduces that blindness in the trained policy regardless of how
well the imitation loss converges.

### 6.2 Exploitation-pull

The inverse-performance weighting in the explore branch (§5.1) closes the
state-blindness gap but introduces a second, independent effect: it pulls
the explore branch's targets toward the same near-optimum region the
exploit branch is already climbing toward. Both branches — via different
routes — now systematically shape targets to cluster around "wherever this
instance's optimum appears to be." The branch *label* still alternates
faithfully with the realized trajectory, but the underlying *geometry* being
imitated under each label converges toward the same pull.

This matters for AUC specifically because good anytime performance needs a
genuinely broad early phase to locate *which basin* the optimum is in,
before narrowing — and BNN prior instances with depth up to 16 can plausibly
have multiple comparable local optima. A policy trained this way is expected
to converge fast and confidently to the *first* decent basin found, and
rarely discover a better, more distant one: helpful on smooth/easy
instances, actively harmful on rugged ones, with no guarantee these net out
favorably once averaged across the instance distribution AUC is computed
over.

There is a subtler way to state what's happening: privileged knowledge of
"where the optimum is" is supposed to stay confined to *constructing the
training target* and never leak into what the policy conditions on. The
policy's input remains honest, but once the target itself is shaped to
always point toward the true optimum's neighborhood, the policy is
effectively being taught "guess where the optimum is from context, then go
there" rather than genuine uncertainty-driven exploration that would still
make sense without ever getting to peek.

**Proposed diagnostic (not yet run):** generate explore-branch targets with
the inverse-performance weighting on vs. off, across many random instances
and early-trajectory states, and measure the spread — distance from current
context, and how often the target lands far from anything already observed.
A visible collapse in spread with the weighting on would make this bias
directly measurable rather than argued about.

### 6.3 Why this can't be fixed by a better per-step oracle

Both gaps above are the same underlying limitation in different clothes: a
per-example, per-step oracle target — however carefully constructed — cannot
represent an object that only exists at the trajectory level. Each
individual correction can look locally defensible; the bias only appears in
aggregate, across many steps and instances. AUC-to-go is a genuinely
different kind of mathematical object from the quantities the oracle can
compute: it is an expectation over everything that happens *after* a
decision, and everything that happens after depends on the very policy being
trained — it cannot be reduced to a fixed differentiable function to descend,
because the function is a moving target defined by the thing being produced.
This circularity is exactly what the Bellman equation / value-function
machinery exists to break, by learning an estimate instead of computing the
quantity exactly.

**Conclusion carried into the staging (§8): privileged search is not a
compromise en route to the "real" method — it is the right first stage,
because it is cheap and gets something training and diagnosable. AUC-level
RL is not an optional refinement; it is the stage capable of seeing and
correcting exactly what per-step imitation is structurally blind to.**

---

## 7. The Feasible RL Extension

### 7.1 Reward and returns

Decompose log-incumbent AUC into a per-step reward:
`r_t = log(incumbent_{t-1}) − log(incumbent_t)`
(the improvement at step *t*; zero on flat steps). Summed over a trajectory,
this telescopes exactly to the total log-incumbent improvement — a per-step
decomposition of the actual quantity being optimized, computable from
**realized self-play outcomes only**, no privileged access required (a real
BO loop observes exactly this).

Return-to-go: `G_t = Σ_{t'≥t} r_t'` (Monte Carlo), or bootstrapped
`r_t + γ V_φ(s_{t+1})` (TD).

### 7.2 Value head — reintroduced for a different reason than originally proposed

The value head was initially motivated by an AlphaZero-style leaf-evaluation
analogy (compare exploit vs. explore branches without paying for a full
rollout). That justification became vestigial once branch assignment became
a hard, deterministic partition (§5.1) rather than a comparison — nothing
was left for the value head to make cheap, and it was correctly cut from the
pure-imitation design.

It re-enters here for two different, real reasons:

1. **Deployment-time AUC-awareness.** Once the oracle is gone at deployment,
   nothing computes an AUC-aware signal any more. A value function trained
   on realized self-play returns is the only device that still works with no
   oracle at inference time — a materially different justification than the
   original leaf-evaluation motivation, and one that survives independently
   of it.
2. **Budget-aware branch/action selection.** Trained via TD or Monte Carlo
   regression toward realized `G_t`, `V_φ(s_t)` can replace the hard,
   budget-blind incumbent/flat partition with an actual comparison — given
   remaining budget, which candidate has higher *estimated* AUC-to-go — which
   directly targets the budget-blindness gap identified in §6.1, rather than
   hoping a generic RL correction happens to fix it.

An additional, concrete reason the value head is expected to help
*mechanically*, not just conceptually: the achievable output range varies
enormously by BNN instance (§2). A raw Monte Carlo return `G_t` would have
correspondingly huge variance across the instance distribution, making any
gradient estimate noisy. `V_φ(s_t)`, conditioned on context, learns to
predict "how much more improvement is plausible from an instance that looks
like this" — an instance-adaptive control variate. This is not hypothetical:
the ActionHead ablation (§4.1) already showed the value output generalizing
well across random instances precisely where the policy output did not,
suggesting this pathway is already positioned to learn exactly this kind of
instance-conditional estimate.

### 7.3 Policy gradient

`∇_θ J = E[A_t · ∇_θ log π_θ(x_t | s_t)]`, with advantage
`A_t = G_t − V_φ(s_t)`. This is the mechanism that actually addresses §6:
a step is credited or penalized by how the *realized subsequent trajectory*
went relative to what was expected from that state — not by how good the
single next observation looked in isolation, which is the exact limitation
identified in both the budget-blindness and exploitation-pull findings.

This is also where the MDN policy head (§4.2) stops being optional: the
policy-gradient update requires a differentiable `log π_θ(x)`, which a
point-regression head cannot provide.

A smaller, more stable first step than full on-policy REINFORCE: keep
oracle-imitation as the base loss, and use the advantage to *reweight* it
(advantage-weighted regression) rather than running full actor-critic RL
immediately — closer to standard SFT-then-RL staging elsewhere, and less
prone to the instability full on-policy policy-gradient training is known
for.

### 7.4 What transfers from AlphaZero, and what doesn't

Walking through AlphaZero's actual mechanism clarifies which parts are
genuinely relevant and which were imported by analogy without checking fit:

- **MCTS + PUCT** balances exploitation of moves that scored well against
  exploration of promising-but-under-visited moves, continuously, via a
  bonus term that shrinks as evidence accumulates.
- **Leaf evaluation** substitutes a learned value/policy for an otherwise
  impossible cheap evaluation of a mid-game position — Go has no way to
  score an unfinished position except by estimating it or playing it out.
- **The self-reinforcing loop** (better network → smarter search → better
  training targets → better network) exists specifically to work around not
  having a cheap ground-truth evaluator.

**We have the thing AlphaZero was approximating.** `prior.evaluate(x)` is a
cheap, exact, differentiable evaluator of any candidate point — equivalent
to being handed a perfect evaluator for free during training. Consequences:

- Leaf evaluation is not needed to make search tractable — our search
  (gradient descent on the true surface) is already cheap and exact; there
  is no expensive rollout to substitute away.
- The self-reinforcing bootstrapping loop is not needed either — our search
  quality is not bottlenecked by apprentice quality the way MCTS's is; more
  random restarts or a better optimizer improve our search directly, not a
  better network.

**What does transfer, stripped of tree-specific machinery**: the core
pattern of running expensive, instance-specific, ground-truth-grounded
search once and distilling its output into a fast, generalizing network — is
exactly Expert Iteration's real contribution, and exactly what the §5
imitation loop already does. PUCT's *soft*, continuously-shrinking
exploration bonus is also worth keeping conceptually in view as a more
principled alternative to the current hard incumbent/flat partition, should
that partition prove too coarse under the diagnostic in §6.2.

---

## 8. Proposed Staging

1. **Multimodality diagnostic (§4.2).** Cheap (~20 minutes), and determines
   whether the MDN is handling a real problem or a self-inflicted one.
2. **Exploitation-pull diagnostic (§6.2).** Equally cheap, and determines
   whether the inverse-performance weighting is silently narrowing
   exploration in a way that would hurt AUC on rugged instances.
3. **Get the full privileged-search imitation loop training end-to-end**
   (self-play rollout → branch labeling → oracle correction → aggregation →
   retrain), tracked against two baselines on held-out instances: random
   search, and a classical acquisition function (EI/UCB) on the same frozen
   PFN's own PPD. This number is not a stopping point to celebrate if
   reached — per §6.3, it is a **measurement of exactly how much AUC is lost
   to budget-blindness and exploitation-pull**, not evidence the method is
   sufficient.
4. **Add the value head and switch the branch decision to a budget-aware
   comparison** (§7.2), warm-started from the Stage 3 policy. Re-measure
   against the same baselines and against Stage 3's own result — the
   relevant comparison is not "beats EI" alone, but "closes the
   budget-blindness / exploitation-pull gap measured in Stages 1–3."
5. **Advantage-weighted / policy-gradient refinement** (§7.3), only once
   Stage 4 is working and measured, using the MDN policy head throughout
   since it becomes structurally required at this stage.

Each stage is scoped to answer one falsifiable question before the next
stage's added complexity is treated as justified, rather than assuming the
full design is necessary and building toward it regardless.

---

## Appendix: Reference Papers and Implementations

| Component | Paper | Reference implementation |
|---|---|---|
| PFN core | Müller, Hollmann, Pineda-Arango, Grabocka, Hutter, "Transformers Can Do Bayesian Inference," ICLR 2022 (arXiv:2112.10510) | github.com/automl/PFNs (maintained; archived predecessor: automl/TransformersCanDoBayesianInference) |
| TabPFN | Hollmann, Müller, Eggensperger, Hutter, ICLR 2023 (arXiv:2207.01848); TabPFN v2: Hollmann et al., *Nature* 2025 | github.com/PriorLabs/TabPFN; minimal educational reimplementation: github.com/automl/nanoTabPFN |
| PFN-for-BO / multi-fidelity | Müller, Feurer, Hollmann, Hutter, "PFNs4BO," ICML 2023 (arXiv:2305.17535); Rakotoarison et al., "IFBO," ICML 2024 | github.com/automl/PFNs4BO; github.com/automl/ifBO |
| VLA | Black et al., "π0," 2024 (arXiv:2410.24164); Physical Intelligence et al., "π0.5," 2025 (arXiv:2504.16054) | github.com/Physical-Intelligence/openpi (official); github.com/allenzren/open-pi-zero (reimplementation) |
| Expert Iteration | Anthony, Tian, Barber, NeurIPS 2017 (arXiv:1705.08439); Anthony, Nishihara, Moritz, Salimans, Schulman, "Policy Gradient Search... without Search Trees," 2019 (arXiv:1904.03646) | github.com/aravinho/hexit (original); no public repo found for the 2019 paper |
| AlphaZero / value head | Silver et al., *Science* 2018 (preprint arXiv:1712.01815) | github.com/suragnair/alpha-zero-general; github.com/blanyal/alpha-zero |
| DAgger | Ross, Gordon, Bagnell, AISTATS 2011 (arXiv:1011.0686) | github.com/Refefer/Dagger |
