# Milestones

Each milestone is independently implementable and independently testable. Read `ROADMAP.md` for
motivation. Do not start a milestone until its dependencies are green.

Conventions:
- `d` = input dim (18), `B` = budget, `t` = context size, `L` = PFN depth, `P` = pool size,
  `K` = candidate set size, `F` = functions per iteration, `G` = siblings per function.
- Everything touching the PFN runs under `no_grad`, bf16, frozen weights.
- Every milestone ships with tests. "Done" means tests pass, not "code exists".

---

## M0 — Environment: BNN prior + ECDF reward

**Deps:** none

**Build**
- `BNNSampler`: sample depth, width, weights from the prior; return a batched callable
  `f(x: [..., d]) -> [...]` on `[0,1]^d`.
- `QuantileTable`: given a sampled `f`, evaluate `N = 1e6` uniform points in one batched forward,
  build the lookup. Store a 4096-point quantile grid **plus the top 1000 values exactly**, not the
  raw 1e6. Interpolate for lookup.
- `reward(f_star) -> g`: `clip(-log10(1 - û), 0, 4) / 4`.
- `BOEnv`: holds `(f, quantile_table, D_t, B)`. `reset(n_init)` seeds with Sobol.
  `step(x) -> (D_{t+1}, g_{t+1}, done)`. Vectorized over a batch of independent functions.

**Do not build:** GPD tail extrapolation. Deferred deliberately.

**Acceptance**
- `g` is monotone non-decreasing within an episode (incumbent is monotone).
- `g ∈ [0,1]` always; `g = 1` exactly when `û > 1 - 1e-4`.
- Quantile-table lookup matches a brute-force lookup on the raw 1e6 samples to <1e-3 in `û` across
  the range, and to <1e-4 in the top 1% (the tail is where precision matters).
- 512 envs step in lockstep with no Python loop over envs.
- Randomly-selected 1e6-point evaluation completes in <1s on the target GPU.

---

## M1 — Frozen PFN wrapper with full-layer KV export

**Deps:** M0 (only for having functions to query; can be built in parallel)

**Build**
- `FrozenPFN.forward(D_t, X_query) -> PFNOutput` where `PFNOutput` carries:
  - `layer_kv`: all `L` layers' train-token hidden states, concatenated into one sequence of
    length `L·t`, plus a learned per-layer embedding added to each block.
  - `bin_logits`: `[n_query, n_bins]` for every query point.
  - `q_final`: final-layer query hidden `[n_query, d_model]`.
  - `y_norm_params`: whatever the PFN's y-transform was fit to on `D_t`.
- Batched over environments with zero padding (all envs at step `t` share context length).

**Acceptance**
- Cost is measurably **linear** in `n_query`: time at `P = 512` is within ~1.2× of
  `512/32 ×` time at `P = 32`. If it's quadratic, query tokens are attending to each other —
  that's a bug.
- `bin_logits` for a given query point are bit-identical whether that point is queried alone or
  inside a pool of 512. This is the query-independence invariant; assert it in a test.
- Permuting `D_t` leaves `bin_logits` unchanged (exchangeability).
- Runs under `no_grad`, bf16, and allocates no activation storage for backward.
- Throughput benchmark logged: tokens/sec at `t ∈ {16, 64, 128}`, `P ∈ {32, 128, 512}`.

---

## M2 — Candidate feature descriptor

**Deps:** M1

**Build**
- `align_bins(bin_logits, y_star, y_norm_params) -> aligned_logits`: resample the bin grid so
  `y*_t` sits at a fixed index. This is the step that makes a conv over the bin axis meaningful.
- `survival(aligned_logits) -> log_S`: log survival function `log P(y' > y)` over the aligned grid.
- `DescriptorEncoder`: 1D conv over the aligned bin axis → 32–64 dim vector.
- `ProbeHead`: linear map from the compressed vector to `(μ, σ, LogEI, PI, q50, q90, q99)`.
  Trained jointly, discarded at inference. This is a *diagnostic*, not a feature path.
- `derived_scalars(bin_logits, y_star)`: exact finite-sum computation of the same quantities from
  the bar distribution, for use as the probe target and as a fallback feature path behind a flag.

**Acceptance**
- `derived_scalars` matches a brute-force MC estimate over the bar distribution to <1e-4.
- `PI = S(y*)` is recoverable from `log_S` by a single index lookup, to numerical precision.
- `EI` is recoverable from `log_S` by a fixed linear functional (verify by fitting a *linear* probe
  to convergence — if a linear map can't recover EI from `log_S`, the alignment is wrong).
- Probe residual on a held-out set is logged and small; regression tests pin it.
- Alignment is invariant: shifting all `y` in `D_t` by a constant leaves the aligned descriptor
  unchanged.

---

## M3 — Candidate pool and deterministic reduction

**Deps:** M1, M2

**Build**
- `propose_pool(D_t, P) -> [P, d]`, fixed mixture:
  - scrambled Sobol, **continued from the episode's global sequence**, not resampled iid
  - Gaussian perturbations of the incumbent at σ ∈ {0.01, 0.05, 0.15}, reflected at the boundary
  - perturbations around the 2nd–4th best observed points
  - exclude already-evaluated points (`f` is deterministic; re-evaluation is waste)
- `reduce_to_k(pool, pfn_out, y_star, K) -> (candidates, source_labels)`:
  top-8 LogEI, top-8 Thompson (one `y` draw per pool point from its PPD, take top-8), top-8
  UCB(β=2), 8 uniform-at-random from the pool. **Deterministic given (pool, PPD, seed).**
- `source_labels` is carried through for instrumentation (M8).

**Acceptance**
- `reduce_to_k` is a pure function: same inputs → same output, no hidden RNG state.
- Sobol continuation verified: coverage (min pairwise distance / discrepancy) improves
  monotonically over an episode; resampling iid does not.
- The random slice is present and never dropped — assert `K_random > 0` in the config validator.
- No candidate coincides with an evaluated point.

> **Why deterministic:** if the reduction were stochastic or learned, `π(a|s)` would be a two-stage
> sampler and the PPO ratio would need a marginal over subset selection — a sum over permutations.
> Determinism here makes `π(a|s)` *exactly* the final softmax, so log-probs and KL are exact.

---

## M4 — Pointwise scorer head

**Deps:** M1, M2

**Build**
- `PolicyExpert`: 2–4 layers, `d_e ≈ 256`. Tokens = `K` candidate tokens + ~4 register tokens +
  1 budget token.
  - Candidate token init: `x_embed(x_k) ⊕ proj(descriptor_k) ⊕ proj(q_final_k)`
  - Each layer: cross-attention into the full `L·t` PFN KV sequence (fresh `W_K`, `W_V` learned
    from the frozen hiddens), then MLP. Pre-norm, residual.
  - **No candidate↔candidate attention.** Candidates attend to registers and to the budget token
    only.
- `GlobalFeatures`: Fourier embedding of `m = B-t+1` and `t/B`; plus `y*_t`, observed-y spread,
  best-minus-second-best gap, y-norm params. Never raw scalars for the budget.
- Scalar logit per candidate. State-conditioned temperature read from the register tokens.

**Acceptance**
- **Permutation equivariance:** permuting the candidate order permutes the logits identically.
  Assert exactly, not approximately.
- **Pointwise invariance:** `s(x_k)` is unchanged when the other `K-1` candidates are replaced.
  This is the property that keeps the future learned-proposal work open — pin it with a test that
  fails loudly.
- `∂s/∂x` is finite and non-zero through the whole stack including the frozen PFN. Gradcheck it.
- Removing the budget token measurably changes the output (guards against silent dropout of the
  conditioning path).
- Privileged-info guard: a unit test asserts `û_t` and any function-level ground truth are not
  reachable from the policy's input pipeline.

---

## M5 — Supervised myopic pretraining

**Deps:** M0, M2, M3, M4

**Build**
- Triple sampler: draw `(f, D_t, x_k)` independently — no rollouts, no sequential dependency.
  `D_t` from random subsets initially, `t` sampled across the full range.
- Target: `g_{t+1}` computed from the ECDF after evaluating `f(x_k)`.
- Regress `ŝ(x_k; D_t)` onto it. Joint with the M2 probe loss.
- DAgger refresh: after RL exists, periodically mix in states drawn from the current policy.

**Acceptance**
- `ŝ` beats LogEI at *predicting* `g_{t+1}` (lower MSE) on held-out prior draws. If it doesn't,
  the descriptor or the read path is broken.
- Greedy-argmax over candidates using `ŝ` beats greedy-argmax using LogEI on episode `Ḡ_1`. This
  is the first real signal that anything was learned.
- Calibration holds across the `t` range, not just in aggregate — bucket by `t` and check.
- Probe residual stays within the M2 regression bounds.

> `ŝ` at convergence is `E[g_{t+1} | D_t, x_k]`: the exact myopic-optimal criterion for our
> objective, not a heuristic. It *is* `Q_1`. RL then only has to buy the non-myopic correction.

---

## M6 — Value expert

**Deps:** M0, M1

**Build**
- `ValueExpert`: register tokens only, cross-attending into the same frozen PFN KV. **No candidate
  tokens.**
- Privileged input path: `û_t`, function-level difficulty statistics, anything else useful. Train
  time only.
- Categorical head: ~51 bins on `[0,1]`, two-hot (HL-Gauss) target, cross-entropy. Not MSE.
- Trained by regression onto MC returns `Ḡ_t`. **Never onto a bootstrapped target.**

**Acceptance**
- `V̄` output is bit-identical under two different candidate-pool draws from the same state. If it
  moves, candidate tokens leaked in.
- Reliability diagram on held-out `Ḡ_t` is near-diagonal.
- Code path assertion: the value target contains no call to `V̄` itself.
- Predictions stay in `[0,1]` by construction.

---

## M7 — PPO loop with group-LOO advantage

**Deps:** M0, M3, M4, M5, M6

**Build**
- Rollout: `F` functions × `G` siblings. Shared ECDF table per function. **Shared init across
  siblings, varied across functions.** Step all `F·G` envs in lockstep.
- Residual parameterization: `s_k = ŝ_myopic(k) + Δ_θ(k)`, `Δ` zero-initialized, `ŝ_myopic` frozen
  from M5.
- Advantage: `A_t^(i) = Ḡ_t^(i) - mean_{j≠i} Ḡ_t^(j)`. Optionally also subtract `V̄`.
- Loss: clipped surrogate with clip-higher (`ε_lo = 0.2`, `ε_hi = 0.28`), entropy bonus with
  `β_H` autotuned against an entropy target, categorical value loss.
- Rollout artifact per state: dataset (~5 KB) + selected `K` candidate coordinates (~2 KB) + old
  log-probs. **Recompute the frozen PFN each PPO epoch**; do not cache KV.

**Acceptance**
- At step 0 of RL, `Ḡ_1` matches the M5 greedy policy to within noise — the residual handoff is
  wired correctly. A jump here means `Δ` isn't zero-initialized or the candidate set differs.
- Ratios are exactly 1.0 on the first inner epoch. Any deviation means the stored candidate set or
  log-probs are stale.
- Group advantage sums to ~0 within each group by construction.
- No target network, no bootstrapped value target anywhere — assert by code inspection test.
- Throughput: end-to-end iterations/sec logged; PFN recompute is <50% of wall clock (if it isn't,
  the batching in M1 is wrong, not the caching).

---

## M8 — Instrumentation

**Deps:** M7 (but write the logging hooks alongside each milestone, not after)

**Log every iteration**
- Fraction of groups with **zero advantage variance** — the saturation monitor. This is the number
  that decides when GPD stops being deferred.
- Policy entropy.
- `argmax π` vs `argmax LogEI` agreement rate.
- `Ḡ_1` gap: policy vs **PFN + LogEI on the same candidate set**. This is the baseline that
  matters — not GP-EI, not random.
- **Chosen candidate's `source_label` as a function of `t/B`.** Sobol-early → local-late is the
  explore/exploit schedule rendered directly. A flat curve means budget conditioning isn't working.
- `V̄` reliability diagram.
- M2 probe residual.

**Acceptance**
- All of the above appear in the run dashboard from the first RL run, not retrofitted.
- The `source_label × t/B` plot is generated automatically per eval.

---

## Dependency graph

```
M0 ─┬─────────────────────────┬── M5 ──┐
    │                         │        │
M1 ─┼── M2 ──┬── M3 ──────────┤        ├── M7 ── M8
    │        │                │        │
    └── M6 ──┘        M4 ─────┘────────┘
```

---

## Explicitly out of scope

Do not build these without a separate decision:

- GPD / peaks-over-threshold tail extrapolation on the reward
- Any TD or `n`-step bootstrapped value target
- Candidate↔candidate self-attention
- Learned or stochastic candidate reduction
- Gradient-ascent or amortized (flow/diffusion) candidate proposal
- Unfreezing the PFN
- Layer-read pruning
- Distilling LogEI as the supervised target (we regress on realized `g_{t+1}` instead)

The candidate feature descriptor (M2) and the candidate selection rule (M3) are both expected to be
revised. Keep their interfaces narrow so that revision is local.