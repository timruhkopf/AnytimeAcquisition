# Learning to Acquire: PFN + ActionHead via Explore/Exploit Expert Iteration

*Complete design rundown — most-likely-to-succeed variant*

---

## 1. Motivation

Standard Bayesian optimization parametrizes the explore–exploit trade-off with
hand-crafted, ahead-of-time heuristics (EI, UCB, entropy search). PFNs already
give a calibrated, in-context Bayesian surrogate essentially for free
(PFNs4BO, IFBO/FT-PFN). The goal here is to stop hand-crafting the acquisition
function on top of that surrogate and instead **learn it end-to-end, directly
against the objective we actually care about: log-incumbent AUC / anytime
regret over the full trajectory.**

The key structural advantage over ordinary RL-for-BO or ordinary Bayesian
active learning: because the synthetic prior is under our control, we have
**privileged information** at training time — the true function, the true
optimum, and (from the frozen PFN's closed-form bar-distribution output) the
exact predictive entropy anywhere. That privileged access is expensive, so it
must be confined to *generating training data*, never to *deployment-time
inference*. This is the same trick AlphaZero-style expert iteration and
Deep Adaptive Design (DAD/iDAD) use to make expensive oracle search pay for
itself only once, at training time. Unlike DAD/iDAD — which need contrastive
*lower bounds* on information gain because they lack ground truth — we have
the exact prior, so oracle targets here are exact, not bounded approximations.

This also sidesteps the two failure modes an earlier RL formulation ran into:

- **Reward sparsity** (lucky early samples make genuine improvements rare) —
  solved by supervising directly against dense, privileged targets instead of
  a sparse regret reward.
- **Long-horizon credit assignment** for exploration — solved by DAgger-style
  on-policy correction (§4), which targets exactly the states the policy
  actually visits and gets wrong, rather than propagating a sparse terminal
  signal backward.

---

## 2. Core components

**1. PFN backbone (frozen).**
Standard IFBO/PFNs4BO-style transformer: train tokens self-attend, test
tokens cross-attend into train-token representations only, bar-distribution
output head trained via NLL on the synthetic multi-fidelity prior. Never
fine-tuned after pretraining — this preserves its calibration guarantee and
is the single load-bearing assumption of the whole design (see §6). The
train-section KV cache is, by construction, a sufficient parametrization of
the PPD at any query point, and is deterministic for a fixed context since
the weights never change.

**2. ActionHead (trained, separate weights).**
A VLA-style expert (mixture-of-experts pattern, à la π0/π0.5) that
cross-attends into the frozen PFN's train-section KV cache — tapping more
than just the final layer, since the final layer is compressed toward the
NLL objective specifically and may attenuate information not needed for PPD
prediction but needed for exploration decisions. Gradients never flow from
the ActionHead back into the PFN (mirrors π0.5-KI, which blocks exactly this
gradient path to protect the backbone's original competence — direct
precedent for freezing here too).

**3. Explicit auxiliary state tokens.**
The PFN's train section is permutation-invariant by design (a calibrated
posterior shouldn't depend on arrival order), so step count, remaining
budget, current incumbent value, and recent-improvement trend are *not*
guaranteed to be recoverable from the cache. These are injected explicitly —
as extra query tokens and/or concatenated into the cross-attention K/V
alongside the PFN cache — rather than assumed to be implicitly present.

**4. Policy + value heads on the ActionHead.**
- *Policy*: proposes the next query location(s).
- *Value*: estimates expected remaining log-incumbent AUC-to-go (bootstrapped,
  AlphaZero-style), used to score corrected branches without full re-rollout
  (§4, step 3).

**5. Privileged oracle / search module** (training-data generation only,
never at deployment):
- *Exploit branch*: a few steps of gradient descent on the **true, known
  prior surface**, started both from the current trajectory point and from
  random restarts, to find a near-optimal next point.
- *Explore branch*: a few steps of gradient descent on the **frozen PFN's
  predictive entropy surface** — closed-form, since the bar-distribution
  output gives entropy as `-Σ p log p` with no Monte Carlo needed — targeting
  entropy reduction at "interesting" points. Interesting points are drawn
  from the *model's own current posterior* (thresholded posterior samples:
  "plausibly better than the incumbent"), **not** from the privileged
  ground-truth optimum. This decoupling is what prevents the explore and
  exploit branches from collapsing onto the same target (see §6).

**6. Branch assignment: partition by realized trajectory outcome, not a
learned rule.**
Steps where the true incumbent changed are labeled exploit-correction steps;
flat steps (no incumbent improvement) are labeled explore-correction steps.
This is a hard, structural partition rather than something the model has to
learn to differentiate, which is the main defense against explore/exploit
collapse.

**7. DAgger-style aggregation buffer.**
Stores raw `(context set of (x, y) pairs, oracle target x*, oracle value y*)`
tuples — **not** KV caches. KV tensors are a function of current weights and
would go stale the moment the ActionHead is retrained; since PFN context is a
set, not sequential state, recomputing the forward pass (and a fresh KV
cache) on demand at training time is cheap and is the correct design, not a
fallback.

---

## 3. Architecture at a glance

```
                     ┌─────────────────────────────┐
 (x1,y1)...(xn,yn) → │   PFN backbone (frozen)      │
   [train tokens]    │   train-train self-attn      │
                     │   → multi-layer KV cache      │
                     └───────────────┬──────────────┘
                                     │ cross-attn (no grad into PFN)
                                     ▼
   step count,        ┌─────────────────────────────┐
   budget remaining,  │        ActionHead             │
   incumbent value, → │  (trained, separate weights)  │
   recent trend       │  cross-attends multi-layer KV │
   [aux tokens]        │  + aux tokens as extra Q/KV   │
                     └───────┬──────────────┬────────┘
                             ▼              ▼
                     ┌───────────┐   ┌───────────┐
                     │  Policy   │   │   Value    │
                     │ (next x)  │   │ (AUC-to-go)│
                     └───────────┘   └───────────┘
```

---

## 4. Optimization strategy: Expert Iteration (EXIT) with explore/exploit search

Round structure, repeated over an infinite stream of synthetic BO problem
instances (each with its own optimum/incumbents):

1. **Self-play rollout.** Run the current PFN+ActionHead policy on a fresh
   batch of sampled instances to generate trajectories.

2. **Branch labeling** at a *subsampled* subset of trajectory points per
   instance (bounded compute; avoids over-optimizing a single instance
   before moving on):
   - realized incumbent-improvement step → **exploit search** (true-prior
     gradient descent) → oracle target `x*`
   - realized flat step → **explore search** (frozen-PFN entropy gradient
     descent at posterior-plausible interesting points) → oracle target `x*`

3. **Branch valuation.** Score each corrected branch with a short, cheap
   rollout (a handful of steps under the true function) plus the current
   **value head's bootstrapped estimate** for whatever trajectory remains —
   the same move MCTS/AlphaZero uses to avoid playing every branch out to
   completion. Value estimates are expected to be biased early on and should
   improve across rounds as the policy improves — that's the point of
   iterating rather than solving this analytically upfront.

4. **Aggregate.** Add `(context, x*, y*)` tuples to the buffer.

5. **Retrain** the ActionHead (policy + value heads) via supervised
   distillation against the buffer — imitation loss on `x*`, regression loss
   on `y*`. The PFN backbone stays frozen throughout.

6. **Repeat**, with new self-play rounds run under the *updated* policy. This
   is what makes it DAgger-style rather than plain behavior cloning: oracle
   corrections increasingly target the states the policy itself visits and
   gets wrong, not a fixed set of stale demonstrations — which is what
   should actually move trajectory-level AUC rather than just next-step
   prediction accuracy.

**Evaluation / stopping criterion:** track log-incumbent AUC on held-out
synthetic instances (later, real AutoML benchmarks) across rounds; stop once
it plateaus.

---

## 5. Why this is the most-likely-to-succeed variant

- **No RL, no sparse reward.** Supervision is direct regression against
  dense, privileged targets — this is what removes the reward-sparsity
  failure mode of the original VLA/RL framing.
- **No probabilistic circuits.** Predictive entropy is already closed-form
  from the bar-distribution head; circuits would solve a problem this design
  doesn't have.
- **Structural, not learned, collapse prevention.** Partitioning by realized
  incumbent status, plus defining "interesting" explore targets from the
  model's own posterior rather than privileged ground truth, is a hard
  decoupling rather than something that has to be discovered during
  training.
- **Privileged cost confined to training-data generation.** The expensive
  search never runs at deployment — same principle as AlphaZero and
  DAD/iDAD, applied to an infinite instance stream rather than one instance
  at a time.
- **Frozen backbone + trained, gradient-blocked action expert has real
  precedent** (π0.5-KI blocks exactly this gradient path; iFlyBot-VLA passes
  VLM KV cache to the action expert directly) — this is a documented VLA
  variant, not a novel unproven pattern.
- **Auxiliary tokens close the one clear information gap** (permutation
  invariance drops step/budget/incumbent history that the PPD itself never
  needed to encode).
- **Value-bootstrapping avoids full-trajectory rollout cost** for every
  correction.
- **DAgger-style on-policy aggregation directly targets trajectory-level
  AUC**, addressing the compounding-error problem that plain imitation would
  hit.

---

## 6. Open risks — what to pilot first

Stated plainly, in the same spirit as the original project doc's honesty
about where the attention-alignment attempt got stuck:

- **KV-cache sufficiency/accessibility for the ActionHead is not guaranteed**
  — it's a reasonable hypothesis with real VLA precedent, but frozen-backbone
  probing experiments in the VLA literature show this can range from
  "unusable" (thin heads) to "partial, limited" (expressive heads). Validate
  early with a linear probe over several PFN layers for the specific
  quantities the ActionHead needs (held-out entropy, distance to incumbent)
  before committing to the full pipeline.
- **How many EXIT rounds until the value-head bootstrap is trustworthy** is
  an empirical question, not something solvable analytically upfront.
- **How shallow the lookahead rollout can be** without losing correction
  signal is likewise empirical.
- **A single gradient nudge per flat step may be too weak** a correction
  signal; may need iterated or multi-step corrections — worth an explicit
  ablation.
- **Synthetic-to-real and low-dim-to-high-dim scaling** — pilot on a small,
  low-dimensional synthetic prior with short horizons and a handful of EXIT
  rounds before scaling toward realistic HPO dimensionality.
