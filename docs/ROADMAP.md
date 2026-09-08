# Learned Acquisition Policy — Discrete Candidate-Set RL — ROADMAP

> **Status:** design frozen, implementation not started.
> **Audience:** implementing agent (Claude Code) + human reviewer.
> **Read `docs/PROBLEM_SETTING.md` first** — the PFN architecture, its
> attention pattern, and why it matters are described there once, not
> repeated here. **Read §1 before writing any code** — these are load-bearing
> decisions from a design discussion, not defaults; reversing them silently
> breaks correctness or trains something that looks fine and isn't.
>
> **Branch history:** this branch (`m4-q-head`) previously carried a
> different design — a `Q(a|D,m)` regression trained via branch-and-replay
> expert iteration, with an exact-DP verification oracle (`oracle/discrete_dp.py`)
> and a privileged-search milestone (`search/`). That work (and its own
> `docs/ROADMAP.md`) is intact on `roadmap-vla-privileged-search` if anything
> here needs to be cross-checked or recovered — it isn't wasted, it's parked.
> This file starts over: a discrete-candidate-set policy trained with
> group-relative Monte Carlo RL, warm-started by distilling `LogEI`.

---

## 0. What this project is

Learn an acquisition policy for Bayesian optimization over a BNN prior
(`docs/PROBLEM_SETTING.md` §PS.1), the same core bet as before — a frozen
PFN as posterior encoder, its own prior as an infinite training
environment — but a different mechanism for turning that into a trained
policy:

**Core loop.** At each decision, sample a candidate set of `K` points from a
fixed, non-learned proposal (`§3 M1`). Score all `K` with a small
cross-attention head reading the frozen PFN's own representation of the
current context. Softmax over the `K` scores gives a categorical policy —
sample (train) or argmax (deploy). Reward per step is the existing
tail-quantile `g`-transform (`reward/tail_quantile_reward.py`, unchanged);
return is the existing budget-normalized `Ḡ_t = (Σ g_s)/m ∈ [0,1]`
(unchanged). Train with group-relative Monte Carlo policy gradient (GRPO-shaped),
warm-started by behavior-cloning onto `LogEI`.

**What's reused, unchanged, from the prior design on this branch:** the BNN
prior (`priors/bnn.py`), the PFN and bar distribution
(`models/pfn.py`/`models/bar_distribution.py`, including `ei`/`pi`/`quantile`/`ucb`),
the frozen-surrogate wrapper (`models/surrogates/pfn_surrogate.py`), the
`g`-reward and `Ḡ_t` return definitions (`reward/tail_quantile_reward.py`),
the rollout/AUC metrics (`metrics/rollout.py`, `metrics/inc_auc.py`), and the
GP/PFN classical baselines (`models/baselines/`). None of this is
mechanism-specific — it's reused because it's genuinely mechanism-agnostic,
not out of inertia.

**What's different:** the policy is discrete-over-a-candidate-set instead of
a value function trained by exact-DP-verified regression; training is
group-relative Monte Carlo RL instead of branch-and-replay expert
iteration; there is no exact-DP verification oracle in this design (the
verification story here is warm-start-vs-LogEI rank correlation and
beat-your-own-surrogate evaluation instead — `§3 M2`/`M4`).

**Formal setting — unchanged.** Still a Bayes-Adaptive MDP with `D_t` as the
(Markov, unordered-set) state (`docs/PROBLEM_SETTING.md` §PS.1); still
minimize throughout.

---

## 1. Invariants — do not change without reading the rationale

### 1.1 The policy scores a sampled candidate set — it does not output a continuous action

The acquisition surface is multimodal by construction (multiple promising
regions at once, especially early). A unimodal continuous policy
(Gaussian or flow, over `[0,1]^d`) averages between modes and lands in the
valley between them — with a noisy critic on top, that's a second thing
that can diverge while the first is still finding its footing. This is a
known, named failure mode for this exact problem class, not a hypothetical.

Instead: draw `K` candidates from a fixed proposal (`§3 M1`), score all `K`
with the cross-attention head, softmax → categorical policy. This gives
exact log-probs, exact entropy, exact KL, no reparameterization variance,
genuine multimodality, and PPO/GRPO clipping works unmodified. **The PFN
already has the right shape for this** (`docs/PROBLEM_SETTING.md` §PS.1):
query tokens cross-attending to train tokens and not to each other *is* a
parallel acquisition-function evaluator over an arbitrary candidate set —
this project already built the architecture for a discrete policy; using
it for a continuous one would be fighting its own shape. **MetaBO and NAP**
(`docs/REFERENCES.md`) are the direct precedent for this parameterization
actually training — cited there, not re-derived here.

Keep the *same* proposal distribution at train and deploy, so the induced
distribution over the cube is consistent between them (mirrors the old
design's own train/deploy-mismatch concern, same underlying reason).
`K` is a real compute knob (`§1.7`) — default `K≈32–64`; spend effort on
proposal *quality*, not on growing `K` further.

### 1.2 Reward and return: unchanged from the prior design, reused deliberately

`reward/tail_quantile_reward.py`'s `g`-transform (log-scale tail-quantile,
percentile against a per-instance dense reference) and `Ḡ_t = (Σ_{s} g_s)/m`
(budget-normalized return, `m` = remaining budget) are reused as-is — this
was already validated (scale-invariance, no-NaN property tests, clip-bind
rate measured) on the prior design and none of that validation is
mechanism-specific. `Ḡ_t ∈ [0,1]` is also, not incidentally, already in
exactly the *return-to-go* form `§3 M0`'s diagnostic needs.

### 1.3 No TD bootstrap in v1 — Monte Carlo with a group leave-one-out baseline instead

The old design's `h`-annealing bootstrap machinery (`V̄` targets, deadly
triad avoidance via slow `h`-annealing) is **not used in v1.** This
environment has a property that makes Monte Carlo strictly better here:
it's cheap, resettable, and the function draw is fully controlled — so pay
for variance reduction with more samples, not with bootstrap bias risk.

**Mechanism — sample `G` episodes (siblings) per function draw**, sharing
the same per-function reference sample (`build_ecdf`, amortized once across
all `G`). Per-timestep leave-one-out baseline:

```
A_t^(i) = Ḡ_t^(i) − (1/(G−1)) Σ_{j≠i} Ḡ_t^(j)
```

This cancels two variance sources at once, not one:
- **Per-function difficulty** — the dominant source, since prior draws vary
  enormously in how findable their optima are. This is the usual reason for
  a group baseline.
- **Reward-estimation noise in the tail.** At `N=10⁶` reference samples, the
  CDF estimate near the `10⁻⁴` tail has ~100 supporting samples, so
  `1−û` carries ~10% relative error there. That error is *fixed per function
  draw* (same reference sample for every sibling), so it's identical across
  all `G` siblings and **cancels exactly** in the leave-one-out difference.
  This is a *specific* argument for group baselines in this exact reward
  design, not just the generic variance-reduction one.

**Add a learned `V̄(s_t, m)` baseline, subtracted in addition** — trained by
regression onto Monte Carlo returns, never used as a bootstrap target. It
captures state-dependence the sibling mean can't (siblings at step `t` are
already in different states by then). Because it's baseline-only, a bad
`V̄` costs variance, never bias — no deadly triad, no target network, no
critic-divergence failure mode to guard against.

**Upgrade path if variance is still too high once this is running:**
VinePPO-style — branch `G` rollouts from the *same* state at a randomly
chosen subset of timesteps, not only from `t=0`. Cheap here specifically
because the environment is a function evaluation, not an LLM rollout — a
real option, not adopted by default in v1.

**Two things not to do, named explicitly because they're tempting:**
- **Max-based bootstrapping** (any DQN/SAC-with-max variant over the
  candidate set). Overestimation bias scales with the number of actions
  maxed over — here that's `K` freshly resampled candidates *every step*,
  which is worse than a fixed discrete action set. If off-policy learning
  is ever added, use expected-SARSA under the current policy, not a max.
- **Analytic gradients through the BNN.** The prior draw is differentiable,
  which makes `∂f/∂x` tempting to use directly. Don't: that's precisely the
  local-ascent signal BO exists to avoid, the incumbent update along the
  way is non-smooth, and long-horizon pathwise gradients through a rugged,
  many-layer tanh landscape are a known instability mode, independent of
  this project's own history with gradient-based search (the *old* design's
  `search/exploit.py`/`search/explore.py`, both dropped from this branch,
  used privileged gradient descent for a genuinely different purpose —
  label generation under a frozen environment, not policy learning — don't
  conflate the two).

### 1.4 Freeze the PFN — train only the cross-attention scoring head

Same freezing argument as the prior design
(`docs/PROBLEM_SETTING.md`-adjacent: a policy that could reshape the PFN
could reshape the landscape used to evaluate it), reinforced by an
external data point: NAP (`docs/REFERENCES.md`) trained a transformer neural
process end-to-end with RL and found it hard enough that a supervised
auxiliary loss was needed just to keep part of the network a valid
probabilistic model. This project already has that valid model,
pretrained; don't risk it. Unfreeze later, if at all, only with a small LR
and an NLL anchor loss to keep the PPD from drifting off-calibration.

### 1.5 The reward must not silently saturate — reinstate the GPD tail now, not later

**Reverses the prior design's explicit v1 decision** (`unclipped_gpd_reward`/`fit_gpd_tail`
were implemented but deliberately left unused, deferred to a later
milestone gated on measuring the clip-bind rate first). That measurement
already happened, on the other branch: **clip-bind rate 0.333** on EI
trajectories — a third of steps already hit the hard `-log10(1-û)=4` cap.
That's the exact DAPO failure mode this design is newly exposed to, and it
isn't hypothetical:

Once every sibling in a group has saturated (`g_t=1` for the rest of the
episode), every sibling's `Ḡ_t` is identical → the leave-one-out advantage
is *identically zero* → that function contributes no gradient at all,
precisely on the functions the policy is already good at, where the
remaining signal (good vs. excellent) is exactly what's left to learn. At
`d≤18` with a many-layer tanh MLP (exploitable structure, plausibly low
effective dimension), a competent policy hitting `B~100` evaluations should
reach the cap *often*, not rarely — this isn't a tail-case worry.

**Fix: activate the already-implemented GPD tail path (peaks-over-threshold
on the top 1%) in v1**, not deferred. If that alone doesn't fully resolve
it, add the fallback too: **instrument the fraction of zero-advantage
groups per batch and drop them** (DAPO's own dynamic-sampling fix) — cheap,
and gives a direct, per-batch health metric either way (`§6`).

### 1.6 Compute/parallelism axes — see `docs/PROBLEM_SETTING.md` §PS.4 for the shape

Three genuinely different costs, not one: within-episode recompute (small
in absolute terms, `K` usually dominates it), across-function batching (the
real parallelism axis — batch `F` functions, not time), and across-gradient-step
recompute (no KV cache needed at all — the state *is* `D_t`, small
enough to store in a replay buffer and recompute the PFN in batch,
embarrassingly parallel, a throughput problem not a latency one). Read
`§PS.4` before optimizing anything here; the naive instinct ("bidirectional
attention means no caching, that's expensive") is right about the caching
and wrong about the expense.

---

## 2. Known traps

### 2.1 The DAPO dead-group problem (see `§1.5` for the fix already mandated)

Worth stating as its own trap because it's easy to reintroduce accidentally
later (e.g. by tightening the reward clip, or training long enough that
even the GPD-tailed reward starts saturating at a *different* threshold):
**any reward with a hard ceiling will eventually produce zero-advantage
groups once the policy gets good enough at a subset of functions**, and a
leave-one-out/group-relative baseline makes that failure *silent* — no
error, no NaN, just a batch that quietly stops teaching the policy anything
about the functions it's already solved. Instrument `§6`'s zero-advantage-group
fraction continuously, not just at v1 launch.

### 2.2 Max-based overestimation and analytic BNN gradients

Both already forbidden in `§1.3`; listed again here because they're the
kind of thing that looks like a reasonable local fix under pressure (a
value-based method feels safer than policy gradient when training is
unstable; a differentiable environment feels wasteful not to use directly).
Neither is safe here for the stated reasons — re-read `§1.3` before reaching
for either.

### 2.3 Unimodal continuous policies collapsing between modes

The concrete failure mode `§1.1`'s decision avoids: a Gaussian/flow policy
trained against a genuinely multimodal target objective doesn't pick a
mode, it averages them, landing in a low-value valley between two good
regions — and looks like an optimization *instability* (loss spikes,
divergence) rather than what it actually is, a parameterization mismatch.
If this project ever revisits a continuous-action policy (e.g. for a
learned proposer, `§7`), re-derive this risk fresh rather than assuming a
different training recipe fixes it.

---

## 3. Milestones

Each milestone has an explicit exit criterion.

- [ ] M0 — RIBBO-style diagnostic: how much headroom above LogEI exists?
- [ ] M1 — Candidate-set + cross-attention scoring head (architecture only)
- [ ] M2 — Warm start: distill `LogEI` onto the categorical head
- [ ] M3 — Group-relative Monte Carlo RL training loop
- [ ] M4 — Evaluation: beat your own surrogate, not just GP-EI/random

### M0 — Cheap diagnostic before committing to RL at all ⛔

**Do this before building any of M1-M3's machinery.** RIBBO
(`docs/REFERENCES.md`) gets a competitive learned BO algorithm with **no
RL**: fit a sequence model to trajectories from a portfolio of behavior
algorithms, each augmented with a return-to-go conditioning token, then
sample conditioned on a high target return at deployment. This project's
own `Ḡ_t ∈ [0,1]` (`§1.2`) is already exactly the return-to-go form this
needs — no new reward machinery.

Generate trajectories from a portfolio (`LogEI`, `UCB` at several `β`,
Thompson sampling from the PFN's own PPD, random, and — once it exists —
the current policy), train a sequence model conditioned on realized
`Ḡ_t`, sample at `Ḡ=1` (or the highest achieved value) at evaluation time.
Optionally iterate — keep the top-quantile trajectories per function,
retrain, repeat (expert iteration; genuinely few failure modes, since
there's no bootstrap and no policy-gradient variance to manage at all).

**Exit criterion:** measure how much headroom above `LogEI` this
return-conditioned model finds, on held-out function draws. **This number
decides whether the rest of this roadmap (M1-M4, real RL) is worth
building.** If the headroom is small, the honest move is to stop and
reconsider, not to proceed to a much more expensive training method chasing
a gap that may not exist. If it's large, M1-M4 have a concrete target to
aim past.

### M1 — Candidate-set proposal + cross-attention scoring head

**No training yet — get the architecture and plumbing correct first.**

`src/anytimeacquisition/search/proposer.py` (new — `search/` was removed
from this branch along with the prior design's privileged-search machinery;
this is a different, much simpler thing: a fixed, non-learned candidate
generator, not a gradient-based oracle search):
- Sobol screen
- local perturbations around the incumbent and a few runners-up
- a handful of Thompson draws sampled from the frozen PFN's own PPD at the
  current context (cheap: the PFN already produces this distribution)
- same proposal at train and deploy (`§1.1`)

`src/anytimeacquisition/models/acquisition/scoring_head.py` (new — the
cross-attention head; naming distinct from the prior design's
`q_head.py`/`Q(a|D,m)` regression target, since this outputs a *score* fed
to a softmax over `K`, not a calibrated value estimate):
- input: the PFN's PPD at each candidate (bar-distribution logits — the
  sufficient statistic, `docs/PROBLEM_SETTING.md` §PS.2) plus a
  cross-attention readout of the frozen context (details TBD against
  `docs/PROBLEM_SETTING.md` §PS.3's open question — how much of the PFN's
  internal state to read is not yet settled, see `§7`)
- budget conditioning **on the policy itself, not just a value function** —
  the optimal acquisition is genuinely non-stationary (explore early,
  exploit late), and without `m` as a policy input there's no way to
  represent that distinction at all. Token or FiLM/adaLN, either is fine;
  this is a smaller decision than *whether* to condition on it.
- output: one scalar score per candidate; softmax over the `K` candidates
  is the categorical policy (`§1.1`).

**Exit criterion:** shapes/gradients check out end to end (forward + backward
through a dummy loss), `m`-conditioning changes the policy's output
distribution when `m` is varied at fixed context (the same sanity check the
prior design used for its own budget conditioning — cheap, and catches a
broken conditioning path before any real training starts).

### M2 — Warm start: distill `LogEI`

`src/anytimeacquisition/trainer/warm_start.py`:
- Behavior-clone the categorical head onto `softmax(LogEI)` over the same
  candidate set (`LogEI` from `models/bar_distribution.py`'s `.ei()`,
  reused unchanged) — no environment interaction, no rollouts, exact and
  cheap.
- Free byproducts: a policy that's already competitive on day one, a
  natural KL anchor for the RL phase (`§3 M3`), and a direct "did RL do
  anything" measurement once M3 runs (compare post-RL vs. this checkpoint).

**Exit criterion:** high rank correlation / low KL between the warm-started
head's ranking and `LogEI`'s own ranking, on held-out contexts — a cheap,
exact architecture sanity check (if the head can't fit a deterministic
closed-form target, something in M1's plumbing is broken, not the RL). The
prior design's `oracle.score_candidate_q`-style rank-correlation tooling was
removed with `M3`/`search/` on this branch — rebuild a minimal version of
just this check here if useful, don't assume it still exists.

### M3 — Group-relative Monte Carlo RL training loop

Implements `§1.3`/`§1.5`/`§1.6` directly:
- `F` functions batched in parallel (hundreds to low thousands, per
  `docs/PROBLEM_SETTING.md` §PS.4 — this is the real throughput axis)
- `G` siblings per function, sharing one reference sample (`build_ecdf`)
  per function
- leave-one-out advantage + learned `V̄(s_t,m)` baseline (`§1.3`)
- reward includes the reinstated GPD tail (`§1.5`)
- PPO-style clipped objective (works unmodified against the exact
  categorical log-probs, `§1.1`), 2-4 epochs per batch, entropy bonus
  annealed over training
- KL anchor to the M2 warm-start checkpoint (or a decaying schedule off it)

**Exit criterion:** trains stably (no collapse in entropy or reward);
zero-advantage-group fraction (`§2.1`) tracked and not silently growing
unaddressed; measurable improvement over the M2 warm-start baseline on
held-out functions, not just a decreasing loss.

### M4 — Evaluation: the bar is your own surrogate, not GP-EI

**Primary baseline: `PFN + LogEI` over the identical candidate set.** If the
trained policy doesn't beat this, the RL isn't contributing anything beyond
what the frozen surrogate already offers for free — this is the number that
actually answers "did this work," ahead of any comparison to a different
surrogate family. `GP+EI` and random search (both already implemented,
`models/baselines/gp_acquisition.py`, `metrics/rollout.py`) stay in the
table too, for external grounding against the prior design's own M2
findings — but they're context, not the pass/fail bar.

**Exit criterion:** full comparison table (policy, `PFN+LogEI`, `GP+EI`,
random) across a spread of `d`/`B`, per-instance normalized (reuse the
per-instance percentile normalization already validated on the prior
design — raw-scale AUC is not directly comparable across heterogeneous
function draws, same reasoning as before, don't relitigate it).

---

## 4. Repository layout

```
src/anytimeacquisition/
  priors/
    bnn.py                       # already exists, unchanged, reused
  models/
    pfn.py                       # already exists, unchanged, reused
    bar_distribution.py          # already exists, unchanged, reused (ei/pi/quantile/ucb)
    surrogates/
      pfn_surrogate.py           # already exists, reused -- may need a return_hidden path (M1, see §7)
    acquisition/
      scoring_head.py            # M1 -- new
    baselines/
      gp_acquisition.py, pfn_acquisition.py   # already exist, reused (M4)
  reward/
    tail_quantile_reward.py      # already exists, reused -- activate the GPD path (§1.5)
  search/
    proposer.py                  # M1 -- new; note: NOT the prior design's search/ (removed)
  trainer/
    warm_start.py                # M2 -- new (different content from the prior design's own warm_start.py)
    grpo_trainer.py               # M3 -- new
  pipelines/
    train_policy.py               # M3 -- new Hydra entry point
  metrics/
    inc_auc.py, rollout.py        # already exist, reused
notebooks/
  m0_return_conditioned_diagnostic.ipynb   # M0
  m2_warm_start_vs_logei.ipynb              # M2
  m4_policy_vs_baselines.ipynb              # M4
docs/
  ROADMAP.md            # this file
  PROBLEM_SETTING.md    # shared background
  REFERENCES.md         # bibliography
```

---

## 5. Config surface

| Key | Default | Notes |
|---|---|---|
| `proposer.K` | `32–64` | candidates per decision (`§1.1`) |
| `proposer.n_thompson` | small | Thompson draws from the PFN's own PPD |
| `train.G` | TBD, tune | siblings per function (`§1.3`) |
| `train.F` | `512–2048` | parallel functions per batch (`§1.6`/§PS.4) |
| `reward.gpd_top_percentile` | `99.0` | already implemented (`fit_gpd_tail`), now active in v1 (`§1.5`) |
| `train.entropy_coef` | anneal | collapse guard |
| `train.kl_anchor_weight` | TBD | anchor to M2 warm start |
| `pfn.freeze` | `true` | `§1.4` — not a v1 knob to ablate casually |

---

## 6. Diagnostics — run continuously, not at the end

| Diagnostic | Catches | Milestone |
|---|---|---|
| Zero-advantage-group fraction | Silent DAPO dead-group failure (`§1.5`/`§2.1`) | M3 |
| Rank correlation / KL vs. `LogEI` | Whether RL moved past warm start at all | M2/M3 |
| Reward clip-bind rate (post-GPD) | Whether the GPD tail is actually resolving the cap, not just moving it | M3 |
| Categorical policy entropy over training | Collapse (over-exploitation) or failure to commit (stuck near-uniform) | M3 |
| `m`-shuffle test | Broken budget-conditioning path (`§3 M1`) | M1 |
| Tokens/sec, wall-clock per step | Whether `F`/`K`/`G` are actually well-batched (`§1.6`) | M3 |

---

## 7. Open questions

1. **How much of the PFN's internal state should `scoring_head.py` read?**
   Only the final PPD (safe, matches "sufficient statistic" argument,
   §PS.2), or also cross-attend into intermediate representations (richer,
   closer to π0.5's own pattern, §PS.3, but mechanically close to the prior
   design's own retired `models/action_head.py` — see that branch's
   history before assuming this is a fresh question). Not settled; resolve
   before or during M1, ablate rather than assume.
2. **Does the VinePPO-style mid-trajectory branching upgrade (`§1.3`) turn
   out to be necessary?** Only answerable once M3 is running and variance
   can actually be measured, not before.
3. **Should the PFN ever be unfrozen (`§1.4`)?** If M4's results plateau
   below what the architecture seems capable of, revisit with a small LR
   and NLL anchor — not a v1 question.
4. **Causal-masked PFN variant, for genuine `O(B)` trajectory KV-caching**
   (mentioned as a live option, not a default, in `docs/PROBLEM_SETTING.md`-adjacent
   discussion): would need a *separate* PFN pretraining run (causal masking
   over randomly permuted train-token order), loses exact exchangeability
   (a real modeling cost, not just an engineering one), but the
   counter-argument — the acquisition-ordered context at decision time is
   already off the prior's own i.i.d.-input measure, so exact
   exchangeability may be less load-bearing at deployment than it sounds —
   is worth a controlled comparison, not an a priori rejection. Explicitly
   **not in scope until `§1.6`'s existing batching plan is shown
   insufficient** — don't build this preemptively.

---

## 8. Explicitly out of scope for v1

- Continuous/flow-based policy over `[0,1]^d` (`§1.1`)
- TD bootstrapping / `h`-annealing value targets, reused from the prior
  design (`§1.3`) — Monte Carlo + group baseline only, in v1
- Max-based (DQN/SAC-style) value learning (`§1.3`/`§2.2`)
- Analytic gradients through the BNN prior for policy learning (`§1.3`/`§2.2`)
- Unfreezing/fine-tuning the PFN (`§1.4`, revisit per `§7.3`)
- Causal-masked PFN retraining for trajectory-level KV-caching (`§7.4`)
- A learned proposer (the proposal distribution in `§3 M1` stays fixed and
  non-learned in v1; a shared-trunk dual-head learned proposer is a
  plausible future extension, not this one)
