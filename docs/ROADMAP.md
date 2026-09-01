# Roadmap

## Goal

Learn an **in-context acquisition function** for PFN surrogates that directly
optimizes the metric BO methods are actually judged on — **anytime
log-incumbent AUC** (area under the log-incumbent curve) — instead of
hand-crafting it on top of a surrogate via EI/UCB/PI/ES-style heuristics.
With few exceptions (e.g. Direct Regret Optimization, which has a different
scope), nothing in the field actually optimizes anytime AUC directly. That's
what this project changes.

Two structural insights drive the approach:

1. **π0.5-style VLA integration.** π0.5 cross-attends an action expert into a
   frozen VLM's KV cache — same backbone, action tokens tapping the vision-
   language representation directly. A PFN's train-section representation is
   exactly analogous: any test token cross-attends into it to get a marginal
   PPD. So an "ActionHead" can cross-attend into a **frozen PFN's train-token
   KV cache** the same way π0.5's action expert cross-attends into the VLM's
   cache — without needing the flow-matching action head π0.5 uses (our
   targets come from a single deterministic oracle per call, not multiple
   human demonstrators, so flow matching's complexity doesn't earn its place
   here).

   *(Note, 2026-08-28: this framing is a simplification, not how π0/π0.5*
   *actually work — their mechanism is a mixture-of-experts joint attention*
   *over concatenated token streams with a shared block-mask, not literally*
   *cross-attention into a separately-frozen KV cache, and their VLM*
   *backbone is fine-tuned throughout training, not frozen. Freezing the*
   *PFN here is a deliberate choice specific to this project, reasoned from*
   *first principles rather than copied from π0.5's actual design —*
   *`docs/log/2026-08-28-pi0-moe-and-frozen-vs-finetuned-backbone.md` has*
   *the full argument, including what the paper does and doesn't actually*
   *justify and the honest cost of freezing.)*
2. **We control the prior, so we can search it directly with GD.** This
   enables Expert Iteration (EXIT) without discrete-decision MCTS. For any
   rolled-out incumbent step: if the incumbent *improved*, run gradient
   descent (multistart) on the **known, differentiable prior surface** to get
   an exploit label. If it *didn't* (a flat/exploration step), run gradient
   descent on the **frozen PFN's closed-form predictive entropy** — optimize
   a new candidate point, appended to context, that reduces entropy at the
   rollout's test tokens / points of interest. Both become imitation-learning
   targets. No RL yet.

This is a **privileged-search-to-imitation** design, not RL, for now — see
[Phase 5](#phase-5--rl-extension-design-only-mapped-out-for-review) for how
RL gets layered on once the imitation loop is working, and why pure
imitation structurally can't fully close the gap to optimal AUC.

## Design reference — read before implementing

The detailed architecture, the full reasoning behind every design choice
below, empirical findings, and a staged validation plan already exist and
are the authoritative spec this roadmap operationalizes:

- `archive/src/exit/PFN_ActionHead_ExpertIteration_Design.md` — the
  most-likely-to-succeed variant, architecture diagram, EXIT round structure.
- `archive/src/exit/Anytime_Acquisition_Claude_Summary.md` — longer version;
  §6 is the specific argument for *why* pure imitation can't fully optimize
  AUC (budget-blindness, exploitation-pull), which motivates Phase 5; §7 is
  the RL extension design.
- `archive/src/exit/claude/pfn-explore-exploit-repo/repo/README.md` — a
  companion prototype implementing pieces of the design, with an honest
  status table (tested vs. scaffolded vs. never run).

When this roadmap and one of those docs seem to disagree, the design docs
win — this file is a project-management view on top of them, not a
replacement.

## Design decisions settled this session

- **No flow-matching action head.** A distributional head instead (already
  the design doc's call, §4.2), gated on the multimodality diagnostic below
  actually being run — see `docs/milestones/M4.md` for the specific choice
  (single vs. mixture of per-dimension Beta distributions, not Gaussian).
- **No discrete-decision MCTS.** `archive/src/exit/train.py` was an earlier
  MCTS-based (`ExpertMCTS`) attempt — it's abandoned and imports files
  (`apprentice.py`, `expert.py`) that no longer exist. Continuous GD search
  over the known differentiable prior (exploit) and the frozen PFN's
  closed-form entropy surface (explore) replaces it entirely.
- **No TabPFN feature-wise attention in the PFN backbone.** TabPFN's
  attention is built for tabular, multi-column data; a single continuous BO
  objective doesn't need it and it "just gets messier." Use a small, custom
  transformer instead — train tokens self-attend, test tokens cross-attend
  into train tokens only, no positional encoding (matches the tested
  `archive/.../repo/src/model/pfn.py`).
- **Do reuse TabPFN's/PFNs4BO's actual bar-distribution implementation** —
  the full one, not the bounded-only reimplementation in the prototype repo.
  It's already vendored at `archive/src/prototype/l2a/utils/bar_distribution.py`
  ("Taken from pfns4BO"). Needs to stay differentiable through the explore
  branch's backward pass (entropy-gradient search), not just usable for the
  forward NLL loss.
- **No RL in the first pass.** Pure imitation against privileged oracle
  targets is the entire first loop (Phases 1–4). RL is Phase 5, mapped out
  for review, not built yet.

## Design decisions from 2026-08-27 discussion

Full reasoning in `docs/milestones/M4.md` and `docs/milestones/M5.md` — summary:

- **Policy head (M4): Beta, not Gaussian, per dimension.** Native `[0, 1]`
  support matches `x`'s domain; also the head M5.5 needs anyway for a
  differentiable `log π(x)`. Single Beta-per-dim if the multimodality
  diagnostic finds no real multimodality, mixture-of-Beta-per-dim if it
  does — multimodality capacity and training-stability/collapse-resistance
  are different axes, so this is a "which variant" choice, not a reason to
  skip the diagnostic. Needs the `α, β ≥ 1` clamp `archive/src/prototype/other_diff/`
  already found necessary (raw values below 1 caused NaN gradients there).
- **Exploitation-pull (M5): fixed structurally, not just diagnosed.**
  Multistart GD's distinct basins (not just the single best point) become
  the explore branch's "interesting points" pool, filtered by plausible
  regret reduction — removes the pull toward exploit's single target by
  construction, rather than relying on the diagnostic to catch it after the
  fact.
- **Trajectory-level inc-AUC weighting (M5): adopted, with mandatory
  per-instance baselining.** Raw inc-AUC weighting would just reward "got an
  easy instance" (achievable output range varies wildly by BNN draw, §2) —
  weight relative to a per-instance baseline instead.
- **Target dimensionality: medium.** Not low — low-dim BO tasks are cheap to
  "solve by luck," which makes baseline comparisons uninformative. Not high —
  out of scope for now. (Exact `x_dim` range is still open, see
  `docs/OPEN_QUESTIONS.md`.) Early correctness/plumbing work (e.g. the
  existing `x_dim=2` checkpoint) still starts low-dim — that's about
  validating the mechanism cheaply, not about the dimensionality baselines
  get compared at.

## Prior art already in `archive/` — read before rebuilding

| Component | Path | Status | Use for this rebuild |
|---|---|---|---|
| BNN prior (vectorized, ECDF-normalized, differentiable) | `archive/src/exit/prior/vectorized_bnn.py` | Written, has a `__main__` smoke check | Primary base for Phase 1 |
| BNN prior (alt, built on l2a's MLP + `pfns4hpo` encoders) | `archive/src/exit/prior/bnn.py`, `archive/src/prototype/l2a/prior/{bnn,mlp}.py` | Written, depends on external `pfns4hpo` pkg | Cross-check against `vectorized_bnn.py`; reconcile into one prior, don't carry both forward |
| Vectorized env wrapper (`vmap`/`functional_call` over a BNN family) | `archive/src/exit/prior/environment.py` (`BatchedTaskFamily`) | Written; referenced by the abandoned MCTS `train.py` | Reference for Phase 1's environment interface |
| PFN transformer (train-train self-attn / test-train cross-attn only, no positional encoding) | `archive/src/exit/claude/pfn-explore-exploit-repo/repo/src/model/pfn.py` | **Tested** — permutation invariance & no test-test leakage verified numerically | Primary base for Phase 2 |
| Bar distribution, bounded reimplementation | `archive/.../repo/src/model/bar_distribution.py` | Tested | Superseded this session — use the row below instead |
| Bar distribution, TabPFN/PFNs4BO original (full) | `archive/src/prototype/l2a/utils/bar_distribution.py` | Vendored | **Use this one** for Phase 2 |
| PFN training loop + trained low-dim checkpoint (`x_dim=2`, 750 steps, CPU) | `archive/.../repo/src/training/train_pfn.py`, `.../runs/pfn_lowdim/` | Tested, actually run | Base for Phase 2; retrain at medium-dim with the TabPFN bar-distribution head |
| Privileged exploit/explore oracle search | `archive/.../repo/src/search/privileged_search.py` | Tested, run against the trained checkpoint | Primary base for Phase 5 |
| ActionHead (VLA-style cross-attention into PFN KV cache) | `archive/.../repo/src/model/action_head.py` | Scaffold — forward pass + PFN-gradient isolation verified numerically, never trained | Primary base for Phase 4 |
| ActionHead demo (`DemoPFN` + `VLAAcquisitionHead`, per-layer K/V projections) | `archive/src/exit/model/asdf.py` | Standalone toy/demo | Reference only — cross-check the per-layer cross-attention approach against `action_head.py` |
| Expert Iteration orchestration loop | `archive/.../repo/src/training/expert_iteration.py` | Scaffold — every step has a working function, runs structurally end-to-end, two placeholders flagged in its own docstring, never trained for real | Primary base for Phase 5 |
| Inc-AUC reward/metric | `archive/src/prototype/l2o_rlsf/model/rewards/area_under_incumbent_curve.py` | Written (`cummin`-based), includes a plotting helper | Base for Phase 3 |
| MCTS-based Expert/Apprentice loop | `archive/src/exit/train.py` (imports missing `apprentice.py`/`expert.py`) | **Abandoned/broken** | Do not resurrect — kept only as a record of a rejected direction |
| Pure BPTT-through-environment meta-learning (Beta head, softmin incumbent) | `archive/src/prototype/other_diff/` | Written, includes an honest self-critique (5 named failure modes: softmin gradient starvation, BPTT shattered gradients, ECDF global-vs-instance minimum, Beta reparameterization instability, "just learning GD" risk) | Superseded — these are exactly the reward-sparsity/instability problems privileged-search imitation avoids |
| PPO / sparse-reward RL formulation | `archive/src/prototype/l2o_rlsf/` (policy, PPO trainer, many reward-shaping attempts, sinusoid env, mlflow callbacks) | Written | Superseded as the *first* approach — reward sparsity & long-horizon credit assignment are exactly what motivated this redesign — but its eval/logging machinery is directly reusable |
| ACQ-token direct local-descent design | `archive/src/prototype/l2a/ROADMAP.md` + `trainer/{acq_trainer,pfn_trainer}.py` | Earlier design notes, predate the finalized EXIT design | Superseded by the EXIT formulation; its budget-aware-PE and multi-fidelity notes are worth re-reading once those phases are reached |

## Phases

Concrete checklist lives in `docs/MILESTONES.md`. Numbering follows the
dependency order (matches how the components were listed when this roadmap
was requested).

### Phase 1 — BNN prior, dual-purpose

Reconcile `vectorized_bnn.py` / `bnn.py` / `environment.py` into one prior
that serves two roles: (a) a differentiable data-generating process to train
the PFN on, (b) a vectorized **environment** — reset under a fresh drawn
parameter set, step by evaluating proposed points — for collecting
explore/exploit rollout alternatives later. Needs to be efficient (batched,
`vmap`-friendly) and support a configurable input dimensionality (the medium-
dim target from above).

### Phase 2 — Frozen PFN checkpoint

A training pipeline that only trains the PFN (custom transformer, no TabPFN
feature-attention, TabPFN's full bar-distribution head) to convergence on
the Phase 1 prior, producing a checkpoint that then never gets fine-tuned
again. Includes the multimodality-adjacent groundwork the design doc flags
before building on top of the checkpoint: verify permutation invariance and
no test-test leakage numerically (as the prototype already did), and check
whether predictive entropy shrinks monotonically with context size at the
target dimensionality (an open, unresolved finding at `x_dim=2`, 750 steps
in the prototype — re-check at medium-dim before trusting the explore
branch's entropy target).

### Phase 3 — Inc-AUC metric

Port/rebuild the log-incumbent AUC computation as a standalone metric
usable for evaluation now and as an RL reward term later (Phase 5). Needs to
work off realized trajectories only (no privileged access) — see the
design doc §7.1 decomposition (`r_t = log(incumbent_{t-1}) − log(incumbent_t)`,
which telescopes to total log-incumbent improvement) for the reward-shaped
version this metric needs to support later.

### Phase 4 — Model: PFN + ActionHead (VLA integration)

The frozen Phase 2 PFN plus a trained ActionHead that cross-attends into its
multi-layer train-token KV cache (tapping more than the final layer),
gradient-blocked into the PFN, with explicit auxiliary tokens for step
count / remaining budget / incumbent value / recent-improvement trend
(permutation invariance means none of this is guaranteed recoverable from
the cache itself). Policy + value heads on top, policy head a Beta
distribution per input dimension (see the 2026-08-27 decision above). Run
the multimodality diagnostic (design doc §4.2/§8 stage 1, ~20 min, not yet
run) — take fixed states from a trained run, call the exploit search with
several random seeds, and check whether resulting targets genuinely cluster
in separate basins (→ mixture of per-dimension Betas) or are tightly
clustered (→ a single per-dimension Beta suffices).

### Phase 5 — Imitation training pipeline (Expert Iteration)

The full EXIT loop: self-play rollout → branch labeling (realized
incumbent-improvement step → exploit search; flat step → explore search) →
aggregate `(context, x*, y*)` tuples into a DAgger-style buffer (raw tuples,
not KV caches — those go stale the moment the ActionHead retrains) → retrain
the ActionHead → repeat under the updated policy. No RL, no value-head
bootstrapping for branch scoring at this stage — branch assignment is a hard
partition by realized outcome, not a comparison (design doc §5.1, §7.2).
Interesting-points selection for the explore branch uses multistart GD's
distinct basins, not just its single best point, so exploitation-pull is
addressed structurally (2026-08-27 decision above) rather than left to the
exploitation-pull diagnostic (design doc §6.2) to merely detect. Imitation
loss examples are additionally weighted by their trajectory's realized
inc-AUC, relative to a per-instance baseline (same decision). Track against
two baselines throughout (Phase 6): random search, and a
classical acquisition function (EI/UCB) run on the same frozen PFN's own
PPD. This isn't a stopping point if it beats those baselines — per the
design doc §6.3, it's a measurement of how much AUC is lost to
budget-blindness and exploitation-pull, which Phase 5.5 (RL) exists to
close.

### Phase 6 — Baselines (medium-dim)

GP + EI/UCB/PI/ES baselines at the same medium dimensionality the EXIT
policy is evaluated at (not low-dim — too easy to solve by luck; not
high-dim — out of scope). Library choice (BoTorch/GPyTorch/scikit-optimize/
other) is still open, see `docs/OPEN_QUESTIONS.md`. These are the reference
points Phase 5's evaluation and Phase 5.5's RL extension both get measured
against.

Other important baselines to implement are random search, but more importantly 
[direct regret optimization in Bayesian optimization](https://arxiv.org/abs/2507.06529)
and [PABBO](https://arxiv.org/abs/2503.00924).

### Phase 5.5 — RL extension (design only, mapped out for review)

**Not implemented yet — this section exists so the design can be reviewed
before any of it gets built.** Once Phase 5 is working and measured, pure
imitation has two structural gaps it cannot close by training longer
(design doc §6): **budget-blindness** (the privileged-search oracle never
sees remaining budget, so geometrically identical states with different
budgets get identical imitation targets) and **exploitation-pull** (the
explore branch's inverse-performance weighting pulls its targets toward the
same near-optimum region the exploit branch already climbs toward, biasing
against genuinely broad early exploration on multi-basin instances). Both
are per-step oracle limitations that only show up in aggregate,
trajectory-level behavior — exactly what a value function and policy
gradient exist to fix.

Proposed staging (design doc §7, §8 stages 4–5):

1. **Reward.** Decompose log-incumbent AUC into a per-step reward
   `r_t = log(incumbent_{t-1}) − log(incumbent_t)` (zero on flat steps),
   computable from realized self-play outcomes only — no privileged access,
   matching what a real BO loop actually observes. This is Phase 3's metric,
   reshaped per-step.
2. **Value head**, reintroduced for two reasons distinct from its original
   (now-cut) AlphaZero-leaf-evaluation motivation: (a) it's the only thing
   that still works with no oracle at deployment time, giving AUC-awareness
   after the privileged search is gone; (b) trained via TD/Monte Carlo
   regression toward realized return-to-go `G_t`, it can replace the hard
   incumbent/flat partition with an actual budget-aware comparison, directly
   targeting budget-blindness.
3. **Policy gradient**, `∇_θ J = E[A_t · ∇_θ log π_θ(x_t | s_t)]` with
   advantage `A_t = G_t − V_φ(s_t)` — this is the mechanism that actually
   corrects both structural gaps, by crediting a step against what the
   *realized subsequent trajectory* did relative to expectation, not by how
   good the single next observation looked in isolation. Requires the MDN
   policy head from Phase 4 (needs a differentiable `log π_θ(x)`, which
   point-regression can't provide) — no longer optional at this stage.
4. **Softer first step than full actor-critic**: keep oracle-imitation as
   the base loss and use the advantage to *reweight* it
   (advantage-weighted regression), closer to standard SFT-then-RL staging,
   before attempting full on-policy REINFORCE.

What does *not* transfer from AlphaZero, and why (design doc §7.4): MCTS
leaf-evaluation exists to substitute a learned estimate for an otherwise
impossible cheap position evaluation — we already have a cheap, exact,
differentiable evaluator (`prior.evaluate`), so leaf evaluation and the
self-reinforcing bootstrap loop AlphaZero needs aren't needed here. What
does transfer is the core Expert Iteration pattern — expensive,
ground-truth-grounded search once, distilled into a fast generalizing
network — which Phase 5 already implements.

### Phase 7 — RealWorld Benchmarking

Once Phases 1–6 exist as real, working components (not scaffolds), We want to check how the incontext 
acquistition capabilities work in the real world. 
