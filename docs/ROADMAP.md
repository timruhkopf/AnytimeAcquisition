# Learned Acquisition Policy over a BNN Prior — ROADMAP

> **Status:** design frozen, implementation not started.
> **Audience:** implementing agent (Claude Code) + human reviewer.
> **Read §0 and §1 before writing any code.** §1 lists decisions that look like
> arbitrary style choices but are load-bearing; reversing them silently breaks
> correctness rather than just performance.

---

## 0. What this project is

Learn an acquisition policy for Bayesian optimization by treating a
prior-data-fitted network (PFN) as a frozen posterior encoder and the PFN's own
prior as an infinite, queryable training environment.

**Core loop.** Sample a function `f` from a BNN prior. That `f` is a free,
callable environment. Run BO episodes against it. Score each episode by an
anytime (AUC-style) objective. Train a small budget-conditioned Q-head to
predict that score for candidate query points. At deployment, encode data with
the frozen PFN, score candidates with the head, query the argmax. The learned
head replaces EI.

**Why it can beat hand-derived acquisition functions.** Four distinct sources,
which must be measured separately because three of them have their own baseline:

| # | Source of gain | Baseline that isolates it | Expected size |
|---|---|---|---|
| 1 | Value transform (improvement measured in ECDF-percentile space, not raw `y`) | `g-EI` | Possibly large, **not attributable to learning** |
| 2 | Marginalizing the unknown per-function value distribution | analytic `g-EI` (not deployable) | Unknown; genuinely novel |
| 3 | Horizon / budget awareness | randomized-horizon PI | **The thesis** |
| 4 | Prior-specific structure (basin shapes, mode counts, decorrelation rate) | all of the above | The "why have a prior" argument |

Sources 2–4 are the project. Source 1 is free and **will contaminate the
headline number if `g-EI` is not in the baseline table from day one.**

**Formal setting.** Bayes-Adaptive MDP with a computable belief. True state is
`(D_t, f)` with `f` hidden, but `p(f | D_t)` is a deterministic function of
`D_t`, so **`D_t` is the state and it is Markov**. No recurrence, no memory, no
history compression. The state is an unordered *set*. Transitions are stochastic
with `p(y | x, D_t)` equal to the posterior predictive.

**Algorithm class.** Approximate policy iteration with a simulator, equivalently
Expert Iteration with the cheapest possible expert (depth-1 branch + rollout).
Not policy gradient, not Q-learning, not actor-critic. See §1.6 for why.

---

## 1. Invariants — do not change without reading the rationale

These are ordered by how badly a violation breaks things.

### 1.1 Ground truth `f` enters ONLY through the training label

`f` must never appear in:
- the policy's inputs,
- action selection during rollout,
- the candidate proposer,
- any feature fed to the Q-head.

Two failure modes:

- **Obvious clairvoyance.** If the label procedure may query `f` while deciding,
  the optimal label is "query the global optimum", which is not a function of
  `D_t` and therefore not learnable. The head regresses toward mush.
- **Subtle clairvoyance.** Labelling each state with the per-function-best action
  and training with cross-entropy is *also* wrong, because
  `E_f[argmax_a Q_f(a)] ≠ argmax_a E_f[Q_f(a)]`. The left side is Thompson
  sampling; the right is Bayes-optimal. The gap is widest when the posterior is
  broad — i.e. early in a run, which is exactly where an AUC objective weights
  improvement most heavily.

**The fix is regression on realized returns.** The L2/CE minimizer converges to
`E[return | D, a, m]`, which is the posterior-marginalized action value. The
regression performs the Bayesian averaging; we never represent a posterior over
`f` explicitly. This is the PFN theorem applied one level up — to the policy
instead of the predictive distribution.

### 1.2 The marginalization mechanism is smoothing, not averaging over repeats

A tempting but **false** justification: "the same `D_t` recurs with different `f`,
so regression averages over them." `D_t` lives in a continuous space of dimension
`t·(d+1)`; you never see the same `D_t` twice.

What actually happens: the loss minimizer converges to `E[target | D, a, m]`
*wherever the function approximator can resolve the conditioning*. The
marginalization is done by the network smoothing across **nearby** `D`.

Three consequences that only follow from the corrected version:

1. **Keep the trainable head small relative to the frozen backbone.** A head
   flexible enough to memorize individual `(D, a)` pairs will fit the single
   realized return rather than the conditional mean, producing a
   Thompson-flavored policy.
2. **The frozen PFN does most of the marginalization work**, because it maps `D`
   into a representation where similar posteriors are nearby. Smoothing there is
   far better conditioned than smoothing in raw `D`-space.
3. **Variance reduction matters more than usual**, because we rely on smoothing
   rather than averaging. Hence common random numbers (§1.9).

### 1.3 Exchangeability: no order-aware encoder over `D_t`

`Q*(a | D, m) = E_{f ~ p(f|D)}[return]`, and `p(f | D)` depends only on `D` as an
unordered set. Arrival order is **irrelevant to the correct answer**.

Worse, order carries information about *which behavior policy generated the
prefix*. With a behavior mixture (§1.8), an order-aware encoder can learn "this
looks like an EI trajectory" and specialize. That is leakage and will not
transfer to deployment.

If the head struggles to compute plateau-like statistics, add them as **explicit
permutation-invariant scalars** (incumbent value, count of observations within ε
of incumbent, number of distinct improvements) — never as an order-aware module.

*This reverses an earlier design draft that proposed a causal trajectory encoder.
Do not reintroduce it.*

### 1.4 Use the frozen PFN exactly as it was trained

Context = `D_t`, queries = candidate points, output = bar distribution over `y`.

**Do not** project `[a, m/B]` into the PFN's hidden dimension and push it through
the PFN's later layers. Those layers were trained to map query tokens of the form
`(x, ?)` to a predictive distribution. A token encoding they have never seen is
out-of-distribution for frozen weights, and the calibration guarantee is lost.

Budget `m` enters the **trainable head only**.

Note: queries attend to context at *every* layer, so there is no intermediate
depth at which "the belief" is fully formed. **There is no belief vector.** What
exists is a function `candidate → predictive distribution`.

### 1.5 Never collapse the candidate representation to `(μ, σ)`

That discards the multimodal, non-Gaussian predictive that is the entire reason
to prefer a PFN over a GP. A policy that sees only two moments is a policy you
could have trained on a GP.

Feed the head the bar distribution re-binned to 32–64 bins, plus derived scalars.

### 1.6 Branch-and-replay, not policy gradient

Policy gradient estimates `∇E[return]` from sampled returns with variance that
scales badly in horizon. We have something PG cannot use: **the ability to
evaluate counterfactual actions at the same state**, because we can branch
against the same fixed `f`. That converts the problem into supervised regression
on a ranking task.

A GP prior could not do this — GP values must be sampled jointly, so each branch
changes the conditioning. **The callable BNN prior is what makes the whole design
possible**, not a convenience.

### 1.7 Continuations must be on-policy; prefixes must not be

- **Prefix** (how we arrived at `D_t`): behavior mixture. Coverage is the goal.
- **Continuation** (what happens after the evaluated action): must follow the
  *current* π, or the target is not `Q^π` and the greedy improvement step is not
  valid policy iteration.

Branching gives both: diverse prefixes, on-policy continuations.

### 1.8 Behavior mixture and warm start are not optional

`argmax Q` with a randomly initialized `Q` visits only uninformative states, so
`Q` never learns what a good state looks like. There is **no self-play curriculum**
here to bail us out — this is the one AlphaGo precondition the setting does not
supply.

Mitigations, both required:
- **Warm start** by regressing returns from EI-generated trajectories. Converts
  policy *discovery* into policy *improvement*.
- **Behavior mixture** throughout: current π + noise, older checkpoints, EI, PI,
  UCB, random search.

### 1.9 Common random numbers

Roll all `K` branched candidates against the **same** `f`, with shared
continuation randomness where possible. Paired comparison rather than independent
samples. Free, and it directly attacks the dominant noise source in what is
fundamentally a ranking problem.

### 1.10 Anneal rollout depth `h` DOWNWARD

- Phase 1: `h = B` — full Monte Carlo, unbiased, **no bootstrap, deadly triad
  structurally absent**. Plain supervised regression on noisy but correct targets.
  Maximally stable while bugs are still being found.
- Phase 2: `h ≈ 8` once `V` is calibrated on validation.
- Phase 3: `h ∈ {1, 3}` for bulk label generation.

The common accident is a fixed small `h` from the start, which leans hardest on
`V` during the exact period when `V` is least trustworthy, and the errors feed
back into the next round of labels.

### 1.11 Reward is absolute; return is horizon-normalized; policy is horizon-conditioned

Do **not** budget-normalize the reward `g_t` itself — it answers "what percentile
is my incumbent at", which has nothing to do with remaining steps, and
normalizing it would make returns non-comparable across episodes.

Budget enters at three other places: the return `Ḡ_t = (Σ g_s)/m`, the policy
input, and the optimal action.

### 1.12 No discount factor

A discount `γ` encodes a single effective horizon `1/(1−γ)` and therefore a single
budget preference — precisely the pathology being removed. The price is a
non-stationary value function, so `m` must be an input and must be **covered** in
training (§4.5).

---

## 2. Known traps

Things that fail silently. Each has a corresponding diagnostic in §6.

### 2.1 `g-EI` is NOT the same as `EI` — the "nothing learnable at depth 1" claim is FALSE

For a raw-value reward, one-step lookahead gives
`m·E[max(f*,y)] = const + m·EI(x)`, so the horizon factor drops out and the
argmax equals EI's. **This derivation does not survive the ECDF reward.**

With `g(·) = clip(−log10(1−F(·)), 0, 4)/4`, the myopic-optimal action maximizes
`E[(g(y) − g(f*))⁺]` — EI computed in `g`-space. Since `g` is a nonlinear monotone
transform that weights deep-tail jumps far more heavily, **this ranks candidates
differently from EI.**

Consequences:
- Myopic `g-EI` already beats EI on this objective, for reasons unrelated to
  horizon awareness. **It is a mandatory baseline.**
- The kill test must hold the transform fixed and vary only the weighting (§3.1).

**Compensating good news:** `g` depends on `f`'s global value distribution, which
is unknown at deployment — you cannot sample a real black box 10^5 times to build
its ECDF. So analytic `g-EI` is **not implementable on a real problem**, while the
learned `Q` is, because it predicts expected `g`-return from `D_t` alone. That is
source #2 in §0.

### 2.2 The `m=1` analytic anchor is APPROXIMATE, not exact

`Q*(a|D,1) = E_{f|D}[ g_f(max(f*, f(a))) ]` and **`g_f` varies with `f`** — each
prior draw has its own ECDF, correlated with the `y` it produces. Summing a
*fixed* `g` against `p(y|a,D)` ignores that correlation.

Use it as a **rank-correlation diagnostic** (expect high Spearman, not 1.0) and as
a **small-weight** auxiliary loss. Do not treat it as exact supervision.

### 2.3 GPD tail produces NaN, and success triggers it

Deferred to M8, but document now. BNN functions with tanh activations on a compact
domain are bounded, so the fitted GPD shape `ξ` is almost always **negative**,
giving a **finite upper endpoint** at `u + σ/|ξ|`. Past it:

```python
gpd_sf = (1 + (xi * x) / sigma) ** (-1.0 / xi)   # base goes negative
```

NumPy returns `nan`; a Python float returns a **complex number**. And
`max(nan, 1e-300)` returns `nan` (the comparison is False), so it propagates
through reward → return → loss → every parameter.

**The frequency rises monotonically with training progress**, because pushing into
the extreme tail is the whole point. Failure is triggered by success.

Required guard when reinstated:

```python
def gpd_survival(x, xi, sigma):
    if abs(xi) < 1e-6:
        return float(np.exp(-x / sigma))
    z = 1.0 + xi * x / sigma
    if z <= 0.0:          # past the finite endpoint (xi < 0)
        return 0.0        # caller MUST clip; -log10(0) = +inf
    return float(z ** (-1.0 / xi))
```

Also wrap `genpareto.fit` in try/except with an empirical-CDF fallback: MLE can
fail to converge, and for `ξ < −0.5` the estimator is not regular (Smith's
condition) — exactly the regime bounded functions occupy.

### 2.4 Reward / target / head must agree on boundedness

A previous draft had three components disagreeing: an unbounded reward, a target
clamped to `[0,1]`, and a `Sigmoid` head that could not exceed 1.0. Net effect:
all deep-tail progress past the clip is invisible to the critic, and the gradient
vanishes at exactly the resolution the tail model was added to provide.

**v1 decision: bounded everywhere.** Clip at 4 decades, `max_score=4`. Boundedness
is not a limitation being worked around — unbounded targets whose variance grows
as the policy improves are the worst possible variance profile for a critic.

### 2.5 `Sigmoid` + `Huber(δ=1.0)` is a double no-op

Sigmoid output is in `[0,1]`, so the max residual is 1.0, so Huber with the default
`δ=1.0` **is identical to MSE** across the entire achievable range — none of the
intended outlier robustness. And sigmoid saturates precisely where targets
concentrate (near 0 early, near 1 late), both flat-gradient regions.

**Use a bar-distribution head with cross-entropy** (§4.3), the same
Riemann-distribution trick the PFN already uses.

### 2.6 The advantage target is undefined at saturation

`Â_t = (Ḡ_t − g_{t−1}) / (1 − g_{t−1})` blows up as `g_{t−1} → 1`.

**Mask those steps out of the loss.** Do not add `ε` to the denominator — that
divides noise by `ε`. Track the mask rate; it is the same instrument that says
whether the GPD is worth reinstating.

Verified properties (safe to rely on): `Ḡ_t ≥ g_{t−1}` always, since the
incumbent is monotone so `g_s ≥ g_{t−1}` for all `s ≥ t`. And
`Q̄ = g_{t−1} + Â·(1 − g_{t−1})` is a positive affine map at fixed state, so the
**argmax is preserved**.

### 2.7 The listwise ranking loss is biased

Softmax does not commute with expectation, so the minimizer of a softmax-CE
against softmaxed realized returns is not the ranking of `Q*` — it is a
Thompson-flavored quantity, a milder version of the §1.1 failure.

Keep **regression as the primary, unbiased loss**. Use listwise only as a
**small-weight, high-temperature** auxiliary (near-linear softmax ⇒ smaller bias),
with common random numbers, and **verify empirically that it helps.**

### 2.8 Ranking ≠ regression

The deployed policy is `argmax_a Q(a)`. What matters is **ordering**, not absolute
values. With a monotone incumbent, most candidates at a state have nearly
identical returns — a network with 1% absolute error can be well-calibrated and
rank almost randomly among the top candidates, which is the only set that matters.

**Primary training diagnostic is top-1 / top-5 ranking accuracy against realized
returns, not MSE.** MSE will look fine while the policy is useless.

### 2.9 Train/deploy action-space mismatch

Labels come from a finite candidate set; deployment uses continuous optimization.
`Q` may be jagged in exactly the directions a gradient optimizer climbs.

Use the **same proposal procedure** in both, and include gradient-ascended-on-`Q`
points among training candidates.

### 2.10 `C` is the expensive knob, and large `C` is useless at high `d`

Cost is **`O(t² + C·t) = O(t(t+C))`** for attention — candidates do not attend to
each other. But at `C ≈ 1024` the **per-token MLP dominates and is linear in `C`**
(roughly 27 GFLOP vs ~0.7 for attention at `t=50`).

And `1024` mostly-random points cover essentially nothing in an 18-dimensional
cube. Use `C ≈ 128–256` at decision points with heavy gradient refinement, and
`C ≈ 64` during rollout *continuations* (where π is merely being executed, not
studied). ~8–16× off the dominant term.

### 2.11 KV cache is intra-step only

Within a step: compute the context KV once, score all `C` candidates against it.
Across steps: **the cache is dead, permanently.** Appending `(x_t, y_t)` changes
every context representation under bidirectional attention, and no fix exists
that preserves exchangeability. Do not build an incremental-cache abstraction.

### 2.12 Environment/surrogate prior mismatch

PFNs4BO's headline model uses the **HEBO+ GP prior**; the BNN-prior model exists
but is not the one they lead with. The environment **must** be the BNN prior
(§1.6). So the prior that makes the best environment may not make the best
surrogate.

**Measure the BNN-prior PFN's standalone BO performance before building on it**
(M2). If substantially weaker than HEBO+, we are constructing a ceiling. v1 uses
the matched prior (surrogate correctly specified w.r.t. environment); mismatched
is a later robustness experiment.

---

## 3. Milestones

Each milestone has an explicit **exit criterion**. Do not start the next
milestone until it is met.

- [x] M0 — Kill test (GO / NO-GO) — see `notebooks/m0_kill_test.ipynb`
- [x] M1 — Environment + reward — clip-bind rate 0.236, see `reward/tail_quantile_reward.py`
- [x] M2 — Frozen surrogate harness + prior sanity check — PFN+EI beats GP+EI at both x_dim 1 & 2 (n=10 envs each), see `notebooks/m2_pfn_surrogate_vs_ei.ipynb`
- [ ] M3 — Exact-DP oracle harness
- [ ] M4 — Q-head + warm start
- [ ] M5 — Branching data generation + core training loop
- [ ] M6 — Stability machinery
- [ ] M7 — Baselines and evaluation
- [ ] M8 — Transfer (this is the actual result)

### M0 — Kill test (GO / NO-GO) ⛔

**~1 day. Everything downstream is conditional on this.**

Exact 2-step lookahead by quadrature on 1-D and 2-D BNN-prior samples,
**holding the reward transform fixed at `g` in both arms** and varying only the
weighting:

- `terminal-g`: maximize `E[g_B]`
- `AUC-g`: maximize `E[Σ_t g_t]`

> Do **not** compare "2-step terminal (raw `y`)" against "2-step AUC (`g`)" —
> that varies transform and weighting simultaneously and the result is
> uninterpretable. See §2.1.

**Exit criterion:** the two policies measurably diverge, **and** `AUC-g` is the
more exploitative. Report the magnitude of divergence as a function of `B`.

**If they barely differ:** the horizon-awareness thesis is dead. The only
remaining content is the value transform, which requires no learned policy.
**Stop and re-scope.**

Deliverables:
- `notebooks/m0_kill_test.ipynb`
- Plot: policy divergence vs `B`; exploitativeness metric vs `B`.
- A written GO/NO-GO paragraph.

---

### M1 — Environment + reward

**Components**

`src/anytimeacquisition/priors/bnn.py` (already exists, extend rather than rebuild)
- Sampler matching PFNs4BO's BNN prior: 8–15 layers, 36–150 hidden units, tanh,
  weights `~ N(0, σ)` with `σ ~ U[0.089, 0.193]`, 14.5% of weights zeroed with
  the remainder rescaled by `(1 − 0.145)^(−1/2)`, pre-activation Gaussian noise,
  input warping.
- Returns a **callable** `f: [0,1]^d → R`, batched, differentiable.
- Fixed seed ⇒ reproducible function.

`src/anytimeacquisition/reward/tail_quantile_reward.py` (already exists, extend)
- `build_ecdf(f, d, N=100_000)` → sorted array. Sobol, not uniform pseudo-random.
  `N=10^5` is chosen for **sufficiency** (5 decades ≥ the 4-decade clip), not
  cost — it is ~13 ms/function on GPU and negligible either way.
- `clipped_empirical_reward(f_t, ecdf, max_score=4.0) -> float` in `[0,1]`.
- `normalized_advantage(G_bar, g_prev, sat_threshold) -> (value, mask)`.
- **Instrumentation (required):** clip-bind rate and advantage-mask rate,
  logged per run.

**Exit criterion — met 2026-09-07, see `tests/test_tail_quantile_reward.py`**
- Reward is scale-invariant: two functions differing by a monotone rescaling
  produce identical reward trajectories for the same query sequence. Assert in a
  test. ✅ `test_reward_is_scale_invariant_under_monotone_rescaling`.
- Clip-bind rate measured on EI trajectories. Record it — it decides M8.
  ✅ **0.236** (23.6%), 8 fresh 2-D BNN draws, 15-step `gp_acquisition_policy(EI)`
  rollout, `n_samples=100_000` Sobol reference (`python -m
  anytimeacquisition.reward.tail_quantile_reward`, seeds fixed — rerun for an
  exact reproduction, this is one seed's reading, not yet averaged over many).
  Non-trivial (neither ~0 nor ~1) — worth a wider seed sweep before trusting it
  as *the* number M8 gates on, but doesn't yet argue either way for
  reinstating the GPD tail.
- No `nan` reachable anywhere in the reward path (property test with adversarial
  inputs). ✅ Found and fixed the exact `_gpd_survival` bug §2.3 predicted (no
  guard existed yet); `test_gpd_survival_handles_past_the_finite_endpoint_without_nan`,
  `test_unclipped_gpd_reward_never_produces_nan_for_adversarial_deep_tail`.

**Known gap, not blocking:** `priors/bnn.py` still doesn't implement input
warping (deliberately shelved per its own docstring, pre-dating this
roadmap) — not exercised by any of the above criteria, revisit if M2's
surrogate-quality comparison or M8's transfer results suggest the prior
itself is the bottleneck rather than the policy.

---

### M2 — Frozen surrogate harness + prior sanity check

`src/anytimeacquisition/models/surrogates/pfn_surrogate.py` (wraps the existing `models/pfn.py`) — **done 2026-09-07**
- Wrapper around **our own** BNN-prior-trained PFN (`models/pfn.py`, checkpoint
  `models/pfn_variable_xdim_smoke.pt`: `d_model=64, n_layers=4` — not literally
  PFNs4BO's own `emsize=512` weights, we don't have those, this is a from-scratch
  model trained on the same *kind* of prior).
- `no_grad`, bf16 autocast, upcast to fp32 before any `BarDistribution` op
  (softmax/log-softmax near ties in bf16 is unreliable) — done via
  `PFNSurrogate.predict()`'s `.float()` before returning logits. Not yet
  verified on `ulysses`'s actual GPU whether bf16 autocast is a real speedup
  there (only run on CPU so far, where it's correct but not necessarily
  faster) — check before assuming it helps.
- API: `PFNSurrogate.predict(x_context, y_context, candidates) -> logits [B,C,n_bins]`.
- Intra-step KV caching only (§2.11) — by construction (one batched forward
  call per whole candidate pool; the PFN's train-side attention never sees
  test tokens), no explicit cache object needed. Do not build a cross-step cache.
- Derived scalars: `PFNSurrogate.expected_improvement`/`.probability_of_improvement`/
  `.mean_std` call straight through to `BarDistribution.ei`/`.pi`/`.mean`/`.variance`.
  **2026-09-08 correction/reversal of this bullet's original wording, by user
  request:** `ei`/`pi` (plus `quantile`/`ucb`, not originally in this bullet
  at all) are now ported directly onto `BarDistribution` itself, from the
  vendored PFNs4BO reference (`archive/src/utils/bar_distribution.py`),
  mirrored for this project's minimize convention — reversing the earlier
  "deliberately dropped, belongs to the classical baselines instead"
  decision. The old standalone `expected_improvement`/`probability_of_improvement`
  functions in `pfn_acquisition.py`/`pfn_surrogate.py` are gone; every call
  site now calls `bar_dist.ei(...)`/`.pi(...)` directly. See
  `bar_distribution.py`'s own docstring for what was and wasn't ported
  (`smoothing`/`mean_prediction_logits` and `FullSupportBarDistribution`'s
  half-normal tail extrapolation were not — this project uses fixed `[0,1]`
  borders, not full support).

**Exit criterion (this is the risk gate for §2.12) — reading updated 2026-09-08,
see `notebooks/m2_pfn_surrogate_vs_ei.ipynb`**
- Standalone BO with this pfn surrogate + EI, benchmarked against a Botorch GP + EI
  model on a shared task set. **Record the gap.**
  ⚠️ **Checkpoint correction (2026-09-08, user-caught):** `pfn_variable_xdim_smoke.pt`'s
  experiment config comment claims "500 steps, not a serious training run" —
  that comment is **stale**. The checkpoint's own logged `history['step']`
  shows it actually trained for **29,999 steps** (301 logged points,
  `log_every=50`... doesn't reconcile exactly with the config's `n_steps:
  500`, so the config was edited/reused after this checkpoint was produced,
  or the checkpoint came from a different run than the committed config
  describes — either way, trust the checkpoint's own logged history over
  the config comment). Not a smoke checkpoint after all.
  - `x_dim=1`, 10 shared environments, 12 steps: PFN+EI **beats** both GP+EI
    and random (mean log-incumbent AUC `-21.70` vs GP's `-19.90`, random's
    `-20.29`; gap `-1.80`).
  - `x_dim=2`, same setup: PFN+EI **beats** GP+EI and random again
    (`-15.35` vs GP's `-15.06`, random's `-14.58`; gap `-0.29`, smaller than
    at `x_dim=1`).
  - Both gaps are small relative to the standard errors (`n=10` environments
    per dimension) — a real, consistent-direction signal (PFN+EI is not
    losing at either tested dimension, unlike the earlier, incorrect
    500-step reading), but `n=10` is still thin. Widen `BATCH_SIZE` and/or
    average over multiple `task_seed`s before treating the exact gap size
    as final.
  - Individual per-environment incumbent curves (not just the mean) are
    plotted in the notebook — worth checking directly rather than only the
    summary numbers, since a mean alone can hide a policy that's a mix of
    great and terrible runs averaging out to something plausible-looking.
- Measured cost curve: PFN forward time vs `t` and vs `C`, confirming the
  `O(t(t+C))` attention term and the `O(C)` MLP term. Use it to pick `C`.
  ✅ Measured (`t∈[4,64]` at `C=64`: 4.5→6.8ms; `C∈[16,256]` at `t=16`:
  4.1→7.1ms, CPU) — near-linear over this range, consistent with fixed
  per-call overhead still dominating the quadratic/attention term at these
  small sizes; re-measure at the larger `t`/`C` M4-M7 will actually use
  before picking `C` from this curve.

---

### M3 — Exact-DP oracle harness

**The only place correctness can be verified rather than merely measured.**

`src/anytimeacquisition/oracle/discrete_dp.py` (new group)
- `d ≤ 2`, grid `k=16` ⇒ 256 actions.
- Exact optimal policy and `Q*` by backward dynamic programming over the belief,
  under the same `g` reward and the same budget-conditioned AUC objective.

**Exit criterion:** for a fixed seed set, `Q*` is computed and cached, and the
harness can score any candidate learned `Q` by rank correlation and by regret
against the DP-optimal policy.

This exists so that later milestones cannot be fooled by a plausible-looking loss
curve.

---

### M4 — Q-head + warm start

`src/anytimeacquisition/models/acquisition/q_head.py` — ~5–10M params, small relative to the frozen backbone
(§1.2).

Per-candidate inputs:
- bar distribution re-binned to 32–64 bins (**never** `(μ,σ)` alone — §1.5)
- EI, PI at 3–4 thresholds, `μ`, `σ`
- the candidate `a` itself
- permutation-invariant context summaries: `g_{t−1}`, `t/B`, count of
  observations within `ε` of the incumbent, number of distinct improvements
- budget `m` by **two redundant paths**:
  1. a token with its own encoder over sinusoidal features of `log2(m)`
     (PFNs4BO's style-embedding mechanism)
  2. **adaLN** modulation of the head's blocks (π0.5's mechanism)
  Ablate to determine which carries the work.

Candidates **do not attend to each other** — mirrors the PFN's query mask, keeps
per-candidate independence (correct for argmax), gives `O(t(t+C))`, and lets `C`
differ between train and deploy.

**Output: bar distribution over `Â ∈ [0,1]`**, ~32 bins, cross-entropy loss.
Take the distribution's **mean** for argmax and for bootstrapping.

`src/anytimeacquisition/search/proposer.py` — identical at train and deploy (§2.9):
1. Sobol screen
2. perturbations around incumbent and runners-up, `σ` shrinking with `m`
3. top-k by EI
4. gradient ascent on `Q` (differentiable in `a`)

`src/anytimeacquisition/trainer/warm_start.py`
- Generate EI trajectories, compute returns, regress `Â`.

**Exit criterion**
- Warm-started `Q` reproduces EI's ranking to high rank correlation on held-out
  states — i.e. it has learned to imitate a known-good policy before being asked
  to improve on one.
- `m`-shuffle test **already passes** at this stage (behavior changes when `m`
  changes) or the conditioning path is broken before RL even starts.

---

### M5 — Branching data generation + core training loop

```
per iteration:
  f ~ BNN prior                        # fresh; NEVER reused
  ecdf ← build_ecdf(f, N=1e5)
  d ~ U{1..18};  B ~ log-uniform[8, 512]

  # prefix from behavior mixture (§1.8)
  roll to a branch point;  m ~ log-uniform      # NOT t ~ uniform (§4.5)

  # branch: K=8..16 candidates, SAME f, shared continuation randomness (§1.9)
  for a in propose(K):
      y = f(a)                         # free
      continue h steps under CURRENT π # on-policy required (§1.7)
      Â[a] = normalized_advantage(...) # masked if g_{t−1} ≥ 1−δ  (§2.6)

  loss = CE(bar_head, Â)                                  # primary, unbiased
       + λ_rank   · high_temperature_listwise_CE(...)     # small; verify (§2.7)
       + λ_anchor · approximate_m1_term(...)              # small; approx (§2.2)
```

Bootstrap under the advantage parameterization:
```
V̄(s_{t+h}) = g_{t+h−1} + Â_θ(s_{t+h}) · (1 − g_{t+h−1})
Ḡ_t^{(h)}  = [ Σ_{s=t}^{t+h−1} g_s + (m − h) · V̄(s_{t+h}) ] / m
```

`h` schedule per §1.10: `B` → `8` → `{1,3}`, annealing **down**.

**Exit criterion**
- Phase 1 (`h = B`, pure Monte Carlo) trains stably with no bootstrap.
- Top-1 / top-5 ranking accuracy on held-out branch points improves over the
  warm-started baseline.
- Ablation run confirming `λ_rank > 0` actually helps (§2.7). If it does not,
  set it to zero and remove the term.

---

### M6 — Stability machinery

- **Replay buffer** over recent iterations.
- **Polyak target network** for the bootstrap.
- **Gating**: promote the rollout policy only after beating the incumbent by a
  margin on a **fixed, seeded** validation suite of held-out prior functions.
  (AlphaGo Zero's 400-game / >55% rule, adapted.) Cheap here — evaluation is just
  BO on cached functions.

**Exit criterion:** a long run (≥ 10× the M5 run length) shows monotone
validation improvement with no oscillation or collapse. Explicitly log and plot
the gating accept/reject sequence.

---

### M7 — Baselines and evaluation

`src/anytimeacquisition/models/baselines/`
- `EI`
- **`g-EI` — MANDATORY (§2.1).** Myopic EI in `g`-space, using the oracle ECDF.
  Not deployable on real problems; included precisely to isolate source #1.
- **randomized-horizon PI** — ifBO's MFPI-random adapted to the single-fidelity
  black-box setting. Hedging over random horizons beat every fixed choice there,
  and acquisition choice mattered at least as much as surrogate quality. This is
  the crude implicit version of what we are learning explicitly, and is the real
  bar for source #3.
- **PFNs4BO's learned KG head** — closest existing learned non-myopic acquisition;
  beat EI on low-dimensional spaces.

**Exit criterion**
- Full table on held-out prior functions.
- **`B`-sweep is mandatory.** AUC at `B=50` and `B=500` can rank methods
  oppositely and both rankings are legitimate.
- **Final regret reported alongside AUC, always.** AUC is what was optimized, so
  it is the one metric that proves nothing.
- Explicit attribution: how much of the gain over EI is `g-EI` (source #1) versus
  the learned policy (sources #2–4).

Expected win regime: small-to-moderate `B`, multimodal functions, early phases of
long runs. Expected null regime: very large `B`, where the horizon weighting
flattens and AUC converges toward terminal regret.

---

### M8 — Transfer (this is the actual result)

Train on the BNN prior; evaluate on **HPO-B, PD1, Bayesmark**.

Precedent is encouraging — PFNs4BO's models were trained purely on prior samples
with no real-data fine-tuning and transferred well — but **a policy is the more
fragile object** and has more room than a surrogate to exploit prior-specific
artifacts.

Infinite *functions* is not infinite *diversity*: the policy cannot overfit
individual functions but can absolutely overfit the prior's idiosyncrasies.

**Optional here, gated on M1's measured clip-bind rate:** reinstate the GPD tail.
Only if the clip binds often. When reinstated, obey §2.3 exactly, keep a clip at
~6 decades, and **raise the head's output range to match** — do not leave the
reward unbounded while the target is clamped (§2.4).

---

## 4. Repository layout

No new top-level package. Per `CLAUDE.md`, every component lives under
`src/anytimeacquisition/<group>/` with a matching `configs/<group>/` Hydra
group; tests mirror `src/`. `priors/bnn.py` and `models/pfn.py` already exist
and are **reused**, not rebuilt — the M1/M2 line items below extend them
rather than starting fresh.

```
src/anytimeacquisition/
  priors/
    bnn.py                       # M1 — already exists: callable, batched, differentiable f sampler
  reward/
    tail_quantile_reward.py      # M1 — already exists: ECDF/GPD tail reward; extend with
                                  #      build_ecdf(Sobol) + normalized_advantage(+mask)
  models/
    pfn.py                       # M2 — already exists: the frozen backbone
    bar_distribution.py          # M2 — already exists: extend with re-binning, EI/PI closed forms
    surrogates/
      pfn_surrogate.py           # M2 — new: frozen wrapper, predict(D_t, candidates) -> BarDistribution,
                                  #      intra-step-only KV cache (§2.11)
    acquisition/
      q_head.py                  # M4 — bar-distribution output over Â
      budget_encoding.py         # M4 — token + adaLN paths
    baselines/
      gp_acquisition.py, pfn_acquisition.py   # already exist
      ei.py, g_ei.py, mfpi_random.py, kg_head.py   # M7 — new
  search/
    exploit.py, explore.py, interesting_points.py  # already exist (prior design; reassess for reuse)
    proposer.py                  # M4 — new: shared train/deploy candidate generation
  trainer/
    pfn_trainer.py                # already exists
    warm_start.py                 # M4 — new
    branch.py                     # M5 — new: K candidates, CRN, on-policy continuation
    targets.py                    # M5 — new: normalized returns, bootstrap
    gating.py                     # M6 — new
    replay.py                     # M6 — new
  pipelines/
    train_pfn.py                  # already exists
    train_q_head.py                # M5 — new Hydra entry point for the branch-and-replay loop
  oracle/
    discrete_dp.py                 # M3 — new group: exact Q* for d ≤ 2
  metrics/
    inc_auc.py                     # already exists
    diagnostics.py                 # §6 — new: ranking accuracy, m-shuffle, behavior leakage, etc.
  benchmarks/
    dummy.py                       # already exists
    hpo_b.py, pd1.py, bayesmark.py # M8 — new
notebooks/
  m0_kill_test.ipynb               # M0 — executed, outputs committed per repo convention
  ...
```

---

## 5. Configuration surface

| Key | Default | Notes |
|---|---|---|
| `reward.max_score` | `4.0` | 4 decades; GPD deferred (§2.4) |
| `reward.ecdf_N` | `100_000` | sufficiency, not cost (§0/M1) |
| `reward.sat_threshold` | `1 − 1e-3` | mask advantage above this (§2.6) |
| `env.d` | `U{1..18}` | sampled per episode |
| `env.B` | `log-uniform[8, 512]` | sampled per episode |
| `train.m_sampling` | `log-uniform` | on `m`, **not** on `t` (§4.5) |
| `train.K` | `8–16` | branched candidates per state |
| `train.h` | `B → 8 → {1,3}` | anneal **down** (§1.10) |
| `infer.C_decision` | `128–256` | (§2.10) |
| `infer.C_rollout` | `64` | continuations only |
| `model.n_bins` | `32` | bar distribution over `Â` |
| `loss.lambda_rank` | small, ablate | biased; must be shown to help (§2.7) |
| `loss.lambda_anchor` | small | approximate (§2.2) |

### 4.5 Budget coverage (called out because it is easy to get wrong)

Sample branch points so that **`m` is roughly log-uniform**, not `t`. Otherwise
mass concentrates at large `m` and the endgame stays undertrained — and the
endgame is where AUC-optimal behavior diverges most sharply from EI.

---

## 6. Diagnostics — run continuously, not at the end

| Diagnostic | Catches | Milestone |
|---|---|---|
| **Top-1 / top-5 ranking accuracy** vs realized returns | The primary failure. MSE looks fine while the policy is useless (§2.8) | M5 |
| **`m`-shuffle**: change `m` at eval, verify behavior changes | Silent total failure of budget conditioning. Likely fix: adaLN, not just the token | M4 |
| **Behavior leakage**: `Q` error on held-out behavior policies (pure random search) vs on-policy | Overfitting to own visitation distribution; order/provenance leakage | M5 |
| **Clip-bind rate** on `g`; **advantage mask rate** | Whether GPD is worth reinstating; whether targets are saturating | M1 |
| **`Q` smoothness** along a dense 1-D slice | Jaggedness in the directions deployment gradient-ascends (§2.9) | M4 |
| **Rank correlation vs `g-EI` at `m=1`** | Broken budget conditioning; expect high, not 1.0 (§2.2) | M4 |
| **Rank correlation vs exact `Q*`** on `d≤2` | Actual correctness | M3 |
| **Gating accept/reject sequence** | Loop drift | M6 |
| **Final regret alongside AUC** | Greedy collapse to a fast plateau | M7 |
| **`nan` / `inf` sentinel on every loss term** | §2.3-class propagation | all |

---

## 7. Open questions

1. **How large is the `g`-transform effect?** If `g-EI` captures most of the gain
   over EI, the learned policy has little room. Only measurable once M7's table
   exists, but it is the top risk.
2. **Is the BNN-prior PFN strong enough as a surrogate?** M2's exit criterion.
   If the gap to HEBO+ is large we are building on a ceiling.
3. **Does `λ_rank > 0` help despite being biased?** M5 ablation.
4. **Which budget-encoding path does the work** — token or adaLN? M4 ablation.
5. **Does the policy transfer, or has it memorized prior artifacts?** M8. This is
   the real result, not a robustness appendix.
6. **Does deeper search pay?** Deferred. Rollout + regression already *is*
   approximate policy iteration with a depth-1 tree. Add tree depth (with
   progressive widening for the continuous action space, using the true `f` as
   simulator at train time) only if an ablation says it pays. Do not start here.
7. **Prior curriculum.** With a frozen surrogate and a fixed prior, the loop
   converges to the best policy expressible under that surrogate's beliefs and
   then stops — approximate policy iteration to a fixed point, not open-ended
   improvement. The missing ingredient is annealing prior hardness (depth, weight
   scale, dimensionality, noise) as the policy improves, substituting for the
   adversary we do not have. **Build the fixed-prior version first, measure the
   plateau, and only add the curriculum if the plateau — rather than the
   surrogate — is the binding constraint.**

---

## 8. Explicitly out of scope for v1

- MCTS / tree search of any depth (see §7.6)
- MuZero-style learned dynamics — the chance-outcome distribution in a stochastic
  environment *is* the posterior predictive, so this would rebuild the PFN with a
  far weaker learning signal (sparse reward through a `K`-step unroll, versus a
  dense per-point likelihood that provably converges)
- Discretizing the input space — `k^d` actions is `10^18` at `d=18, k=10`, the
  state tensor is exponential in `d`, and discretization destroys the smoothness
  that makes surrogate generalization work
- Unfreezing the PFN — doing so makes the head's input distribution non-stationary
  simultaneously with the moving label, which is materially harder to stabilize.
  Legitimate later experiment at ~10× smaller LR; not a default.
- GPD tail model (gated on M1's clip-bind rate; see M8)
- Multi-fidelity / freeze-thaw extensions
