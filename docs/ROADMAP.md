# Roadmap — learned acquisition policy on a frozen PFN

## 0. What this is

We have a Prior-Data Fitted Network (PFN) trained as a surrogate on a BNN prior. It samples
depth, width and weights to generate a function, then learns to produce a valid posterior
predictive distribution (PPD) over `y` at query locations given a train context, decoded as a
binned bar distribution and trained by NLL against `BNN(x) = y`.

Architecturally: train tokens self-attend bidirectionally; query tokens cross-attend to train
tokens only (never to each other), using the same MHA weights. Repeated over `L` layers, then a
linear decoder to bin logits.

The consequence we build on: the train context is a *sufficient state* for the optimization
problem, and that state is spread across the layers of the PFN but is fully accessible by
cross-attention. The query-token independence property means we can score an arbitrary number of
locations in a single forward pass at cost linear in the number of queries.

**Goal.** Stop using a hand-written acquisition function on top of the PPD. Learn an acquisition
policy directly, by playing episodes on the BNN prior, minimizing the AUC of the log-incumbent
curve.

---

## 1. Objective and reward

### 1.1 Why not raw regret

`argmin AUC = argmax Σ_{t=1..B} f_t*`, so the policy objective needs no `y*`. But the training
label does need normalization or the loss is dominated by prior draws with large outputscale. And
estimating `y*` on a 15-layer tanh MLP in `d=18` by multi-start gradient ascent systematically
undershoots, injecting per-function label bias that looks exactly like reward noise and is worst
on the hardest functions. That is the wrong failure mode: labels least accurate precisely where
the policy needs to be smartest.

### 1.2 The reward we use

Score the incumbent by its tail quantile under the input distribution:

1. Draw `N = 1e6` points uniformly in `[0,1]^d`, evaluate the sampled BNN in one batched forward
   pass. Once per function, amortized over the episode.
2. `û_t` = empirical CDF at `f_t*`.
3. `g_t = clip(-log10(1 - û_t), 0, 4) / 4  ∈ [0,1]`
4. `G_t = Σ_{s=t..B} g_s`; objective = maximize `E[G_1]`.

Four properties this is defended on:

- **No `y*` needed.** Removes the systematic label bias entirely.
- **Invariant to monotone transforms of `f`.** Prior draws with wildly different output scales
  become commensurable for free.
- **Log-scaled**, so the deep tail counts: 99th → 99.9th is worth as much as 90th → 99th.
- **Bounded.** `G_t ∈ [0, B-t+1]`, no heavy tail. Log-regret is unbounded below and its variance
  explodes exactly when the policy is doing well — the worst possible variance profile.

### 1.3 Horizon normalization

`V` predicts the horizon-normalized return `Ḡ_t = G_t / (B-t+1) ∈ [0,1]`, `m = B-t+1`.

The raw sum spans two orders of magnitude across `(t, B)` pairs, so a head trained on it spends
most of its capacity learning a trivial multiplicative scaling and its gradients are
heteroscedastic by a factor of `B`. Normalizing puts every target in `[0,1]` regardless of budget;
recover the sum by multiplying.

### 1.4 Known limitation: reward saturation

The clip at `-log10(1-û) = 4` binds at the top 100 of 1e6 samples. A competent policy in `d=18`
will reach that. When it does, all remaining `g_t = 1`, the group advantage is identically zero,
and that function contributes no gradient — the homogeneous-group failure mode, arriving exactly
on the functions where we most want signal.

**Accepted for now.** GPD (peaks-over-threshold) tail extrapolation is the fix and is deferred.
The mitigation in the meantime is *instrumentation*: log the fraction of groups with zero
advantage variance every iteration. That number tells us when this stops being acceptable.

---

## 2. Why not TD

Targets are pure Monte Carlo: `Ḡ_t = (1/m) Σ_{s=t..B} g_s` from the realized rollout. `V̄` is
trained by regression onto that, and appears **only in the advantage, never in its own target**.

Rationale: episodes are ~1e2 steps and the simulator is free and resettable — the two conditions
under which MC beats TD. No bootstrap chain, no target network, no deadly triad. A badly
calibrated `V̄` costs variance, never bias. Revisit only if `B` grows into the thousands.

The dominant variance source is per-function difficulty, and we kill it with a group baseline
rather than a critic:

```
A_t^(i) = Ḡ_t^(i) - (1 / (G-1)) Σ_{j≠i} Ḡ_t^(j)
```

`G` sibling episodes per function draw, sharing the ECDF table. Leave-one-out keeps it unbiased.

**Second, less obvious benefit:** with `N = 1e6`, the CDF estimate near the 1e-4 tail has ~100
supporting samples, so `1 - û` carries ~10% relative error. That error is *fixed per function
draw*, so it appears identically in all `G` siblings and subtracts out exactly. Reward-estimation
noise becomes a common-mode term. This is a real argument for group baselines here beyond the
usual one.

Share the initial design across siblings, vary it across functions: siblings then diverge only
through policy stochasticity, which tightens the baseline.

`V̄` is optional for v1.

---

## 3. Architecture

### 3.1 Frozen PFN

The PFN is frozen. All forwards under `no_grad`, bf16. This is most of the memory budget back,
and it protects PPD calibration from gradient signal originating in a saturating reward.

Per env step:

- **Train tokens:** the `t` observed pairs, encoded exactly as during PFN pretraining. The
  y-normalization (`BarDistribution`'s bins, `BNNPrior`'s ECDF) is **fixed** — set once, never
  refit per episode or per step (verified against the actual implementation, 2026-09-08; an
  earlier version of this doc claimed it was refit to `D_t` every step, which doesn't match the
  code and is corrected here). What DOES move over an episode is the incumbent `y*_t` within that
  fixed representation — see 3.3.
- **Query tokens:** the whole candidate pool `P` at once, `x_encoder(x*) + [MASK]` in the y slot.
  Cost is linear in `P` because query tokens don't attend to each other. `P = 512` is not
  meaningfully more expensive than `P = 32`.

Retained from the forward: per-layer train-token hidden states `H^ℓ ∈ R^{t×d}` for **all** layers,
and the bar-distribution logits for every pool point.

### 3.2 Full-layer read

Read all `L` layers. Do not depth-match expert layers to PFN layers. Concatenate all layers'
train-token hiddens into one KV sequence of length `L·t`, add a learned per-layer embedding, and
let a shallow (2–4 layer) expert attend into the whole thing.

With `t ≤ 100` and `L = 12` that is ~1200 keys — negligible. This gives the full read with a
shallow expert, and layer pruning later becomes an attention-mass measurement rather than an
architecture change.

Learn fresh `W_K`, `W_V` from the frozen hiddens rather than reusing the PFN's own projections —
those were shaped for the PPD objective, and the extra projection over `t` tokens costs nothing.

### 3.3 Candidate feature descriptor

**Target representation: location + raw (compressed) distribution at that location.** Derived
scalars (UCB, PI, LogEI) are a bootstrapping convenience, not the destination — the raw
distribution plus the location is what lets the model extract anything we might want, including
criteria we haven't thought of.

Two preprocessing choices make the raw representation strictly better than derived scalars rather
than just more general:

**Fixed representation, explicit incumbent conditioning — not bin re-alignment.** An earlier
version of this section proposed resampling the bin grid every step so `y*_t` sits at a fixed
index, reasoning that the PFN's y-transform is refit to `D_t` per step. That premise is wrong (see
3.1) — the bins and the y-normalization are both fixed for the life of an episode. Re-indexing them
anyway would still be a bad idea even if the premise held: it makes "bin `k`" mean something
different at every `t` (wherever the incumbent happens to be), so a scorer training across many
different `t` never sees a stable mapping to learn from, and it bakes "distance from incumbent"
directly into the input's structure — which makes one-step-improvement prediction trivially
readable off the input rather than a genuine test of what the model extracted (flagged directly by
user review, 2026-09-08).

Instead: keep the representation exactly as fixed as it already is, and condition on the incumbent
*explicitly* — e.g. append `y*_t` itself (or the log-survival function evaluated at `y*_t`, since
PI/EI are functionals evaluated there) as an extra feature/embedding, rather than transforming the
axis. Every acquisition criterion is still a functional of `p_k` *relative to the incumbent*; the
model gets what it needs to compute that from an explicit conditioning signal, not from a moving
target dressed up as a fixed one. (Prototyped as a per-layer learned incumbent-anchor token in
`models/layer_locked_readout.py`'s `LayerLockedReadout`, built for the §3.2 readout experiment —
the same mechanism generalizes to this section's descriptor.)

**Feed the log survival function, not the pdf.** `S_k(y) = P(y' > y)`. Then `PI = S_k(y*)` is a
single index lookup and `EI = ∫_{y*}^∞ S_k` is a fixed linear functional; UCB is a quantile
lookup. All three land in the span of a linear head on `S`. "The sensible acquisitions are
reachable in one step" becomes a property of the representation rather than hardcoded features.

**Compression:** 1D conv over the aligned bin axis down to ~32–64 dims. Add an auxiliary probe
loss — a linear map from the compressed vector to `(μ, σ, LogEI, PI, quantiles)`, trained jointly
and then discarded. This keeps the guarantee that nothing important was squeezed out, without
hardcoding.

Candidate token init: `x_embed(x*_k) ⊕ proj(compressed_S_k) ⊕ proj(q^L_k)`.

### 3.4 Pointwise scorer — no candidate self-attention

Candidates do **not** attend to each other. The scorer is `s_θ(x; D_t)`, a scalar field on the
domain.

This is the load-bearing decision for the whole future direction. A pointwise scorer can be handed
to an inner optimizer, differentiated in `x`, and maximized — which is the route out of
heuristically proposed candidates (§5). A set-attending scorer is a function of the pool
realization and nothing else. Self-attention would buy joint/batch selection and non-max
suppression, neither of which we need for single-point acquisition.

The one real thing it would have bought — adapting softmax peakiness to how good the available set
is — we get from a **state-conditioned temperature** read off the register/budget tokens. Cheaper,
and it doesn't break differentiability in `x`.

### 3.5 Global conditioning

Per env, not per candidate: remaining budget `m = B-t+1` and `t/B` under a Fourier embedding
(never raw scalars), `y*_t`, observed-y spread, gap between best and second-best, and the
y-normalization parameters.

The budget token is not optional. The optimal acquisition is non-stationary — explore early,
exploit late — and without `m` in the *policy* input (not just the value input) there is no way to
represent that.

**Privileged-information rule:** the policy sees only what is computable from `D_t`. Never feed it
`û_t`. The ECDF is simulator information that will not exist at deployment; a policy conditioned
on its own true percentile will fail on real problems. `V̄` is discarded at test time, so it gets
everything — `û_t`, realized function difficulty, whatever helps. Asymmetric actor-critic, free
variance reduction.

### 3.6 Value expert

Separate module: register tokens only, cross-attending into the same frozen PFN KV, no candidate
tokens at all.

- `V̄` must be a function of state, not of the candidate draw. If it reads candidate tokens the
  baseline inherits noise from the random proposal — injecting variance into the thing whose job
  is removing it.
- It lets us feed privileged features to `V̄` with zero risk of leakage into `π`.

**Categorical head, not regression.** `Ḡ_t ∈ [0,1]` is bounded: discretize into ~51 bins, two-hot
(HL-Gauss style) target, cross-entropy loss. Categorical value heads are consistently more stable
than MSE, and we already have the binned-decoder pattern in the codebase from the PFN itself.

---

## 4. Training

### 4.1 Phase 1 — supervised myopic pretraining

Do **not** distill LogEI. We have the BNN, so regress the candidate score onto the **realized
one-step reward increment**: for state `D_t` and candidate `x_k`, evaluate `f(x_k)`, compute
`g_{t+1}` from the ECDF, regress `ŝ(x_k; D_t)` onto it.

The minimizer of that regression is `E[g_{t+1} | D_t, x_k]` — the exact myopic-optimal criterion
for *our* objective. Not EI, which is a heuristic; not the PFN's EI, which is a heuristic computed
from an approximate posterior. Strictly better teacher, and free.

Three consequences:

- **The supervised head is literally `Q_1`.** Horizon-1 Q-learning by regression. RL then only has
  to buy the non-myopic correction, which is a far smaller thing to discover than an acquisition
  function from scratch.
- **No rollouts needed.** Sample `(f, D_t, x_k)` triples independently and regress. Fully
  parallel, no sequential dependency, enormously cheaper per sample than the RL loop.
- This phase is also what teaches the model to *interpret* the location and the raw distribution
  at that location, which is the point of §3.3.

**State distribution caveat.** If `D_t` comes only from random subsets, we are training off the
state distribution the policy will actually induce. Refresh periodically with states from the
current policy (DAgger-style), or the myopic head will be miscalibrated exactly in the exploited,
narrow-spread regime where it matters.

### 4.2 Phase 2 — RL, residual handoff

Parameterize the RL-time logit as:

```
s_k = ŝ_myopic(k) + Δ_θ(k)          Δ zero-initialized
```

An architectural anchor rather than a KL penalty. We start exactly at myopic-optimal and RL can
only add. Better behaved than annealing `β_KL`, and one fewer coefficient to tune.

Loss:

```
L = -E[ min(r_t A_t, clip(r_t, 1-ε_lo, 1+ε_hi) A_t) ]
    - β_H H(π_t)
    + c_V CE(V̄, twohot(Ḡ_t))
```

`r_t` is per-step and there is exactly one action per step, so the token-level vs sequence-level
importance-sampling debate does not arise — this is a plain MDP. Use clip-higher
(`ε_hi ≈ 0.28`, `ε_lo ≈ 0.2`); acquisition policies collapse to greedy fast and asymmetric
clipping is the cheapest counterweight. Autotune `β_H` against an entropy target rather than
fixing it.

### 4.3 Rollout loop

Per iteration:

1. Draw `F` functions from the prior. Evaluate `N = 1e6` uniform points each and build the
   quantile lookup. Do not store 1e6 floats × `F` — store a 4096-point quantile grid plus the top
   ~1000 values exactly (tail resolution is the only place precision is needed) and interpolate.
2. Initialize with `n_init ≈ 2d` Sobol points. Shared across siblings, varied across functions.
3. Replicate to `F×G` parallel envs, step in lockstep. Every env at step `t` has the same context
   length, so this batches with zero padding: one PFN forward over `F·G × (t + P)` tokens per
   step. The sequential axis is only `B` long; the batch axis is whatever fills the GPU.
4. Rewards, `Ḡ_t`, group-LOO advantage, PPO update on the trainable experts only.

**PPO epochs: do not cache KV.** Store the dataset (~5 KB) and the selected `K` candidate
coordinates (~2 KB) per state, and recompute the frozen PFN each epoch in a big batch. It is
no-grad bf16 over a frozen model — recomputation is cheaper than the memory traffic of a KV store,
and storing candidates explicitly is what guarantees the ratio denominators stay valid across
epochs.

The "bidirectional attention means we can't KV-cache" concern is real but narrowly scoped: it
costs `O(B²)` token-forwards per episode instead of `O(B)`, on an absolutely tiny number. It is
fixed by batching across environments, not by caching.

---

## 5. Deferred, in rough priority order

**Learned candidate proposal.** The scorer is pointwise and the PFN is differentiable in the query
location, so `∂s_θ/∂x` exists through the whole stack. The honest version of BO's inner loop:
Sobol restarts → gradient ascent on `s_θ` → converged points are the candidate set → softmax over
them. Sobol demotes from "action set" to "restart set", which is what it is in every classical BO
implementation.

> Correctness detail: if the candidate set depends on `θ`, then `π_old` and `π_θ` are defined over
> *different* sets and the PPO ratio is meaningless. Run the inner ascent under `θ_old` — the
> rollout snapshot — and score the resulting fixed set under `θ`. The candidate set becomes part of
> the frozen rollout artifact stored alongside the dataset.

The fully amortized version (conditional flow or diffusion proposal trained toward the Boltzmann
distribution of `s_θ`) gives exact continuous log-probs and handles multimodality properly, but
it's a second trainable stochastic thing that can diverge. Gradient ascent gets most of the
benefit with none of that.

**GPD tail on the reward.** Peaks-over-threshold fit to the top 1%, extrapolate the CDF past 1e-4.
Removes the saturation dead zone. Triggered by the zero-variance-group monitor.

**Candidate feature descriptor revision.** Expected. The §3.3 design is a starting point.

**Candidate selection rule revision.** Expected, and largely subsumed by the learned proposal.

**Layer-read pruning.** Only after measuring attention mass per layer.

**Unfreezing the PFN.** Last, if at all. Requires an NLL anchor and a non-saturating reward first.

**Causal-masked PFN for trajectory-level KV-caching (optional; not a default direction).**
§4.3 already settles v1's cost story — batching across `F·G` envs, not caching, fixes the
bidirectional recompute cost, and that's the current plan. This is here because it was actually
tested empirically (branch `roadmap-vla-privileged-search`, `docs/logs/2026-09-08-causal-vs-bidirectional-pfn-comparison.md`),
not proposed speculatively: a causal-masked PFN (lower-triangular train-train self-attention,
retrained from scratch) trades exact exchangeability for a real prefix-caching property (a train
token's hidden state at every layer is provably unchanged by later tokens — verified directly, not
assumed), which would let a rollout extend a KV cache incrementally instead of recomputing `O(t)`
per step.

The measured cost of that tradeoff, at matched architecture (`d_model=64, n_layers=4`) and prior
(`x_dim=6`, variable-dim), causal given **2x** the bidirectional checkpoint's training budget
(59,999 vs. 29,999 steps) specifically to give it a fair chance: causal loses on held-out NLL at
every tested dimension 1–6, though the gap is small (order 0.02–0.19 NLL depending on dimension,
overlapping standard errors at some dimensions). The learning-curve extrapolation (binned
`NLL(step) = a + b/step` fit, R²=0.89) projects **no crossing** — causal's own fitted asymptote
sits ~0.047 NLL worse than bidirectional's plateau, not "closes with more steps." Read that
extrapolation as suggestive, not proven (one run per architecture, projected ~2.5x past the
observed range) — but it agrees with the direct comparison, not contradicts it.

**Why this stays optional rather than becoming a plan:** the gap is small enough that this is a
real, defensible tradeoff to keep on the table — a modest, fairly consistent quality cost against a
throughput benefit this design doesn't currently need (§4.3's batching already fixes the cost this
would address). Revisit only if M7's own instrumentation (`Ḡ_1` gap tracked against wall clock, not
just against the LogEI baseline) shows PFN recompute genuinely dominating iteration time at a scale
`F·G` batching can't absorb — not before, and not by default. If revisited, retrain at whatever
architecture scale (`L`, `d_model`) the rest of this roadmap actually lands on, not at the smoke
scale this comparison used — the gap's exact size at `L=12` is unmeasured.

---

## 6. Instrumentation (from day one, not bolted on)

| Metric | Why |
|---|---|
| Fraction of groups with zero advantage variance | The saturation monitor. Tells us when the deferred GPD stops being deferrable. |
| Policy entropy | Acquisition policies collapse to greedy; this is the early warning. |
| `argmax π` vs `argmax LogEI` agreement rate | "Is RL doing anything" |
| `Ḡ` gap: policy vs PFN+LogEI on the same candidate set | The only baseline that matters. Not GP-EI, not random. |
| Chosen candidate's proposal branch vs `t/B` | Renders the explore/exploit schedule directly. If flat, budget conditioning isn't working. |
| `V̄` calibration (reliability diagram on `Ḡ`) | Cheap, catches a broken value head immediately. |
| Probe-loss residual on the compressed descriptor | Catches over-compression of the bar distribution. |
