# ActionHead search-depth options: single-shot readout vs. amortized argmax-finding

**Date:** 2026-09-02
**Related:** `src/anytimeacquisition/models/action_head.py` (module docstring,
`docs/milestones/M4.md`'s single-vs-mixture gate), `docs/log/2026-08-28-pi0-moe-and-frozen-vs-finetuned-backbone.md`,
`docs/log/2026-09-01-explore-fallback-and-credit-assignment-open-questions.md`

## Motivation / hypothesis

While comparing `ActionHead` to pi0/pi0.5's actual attention mechanics
(joint block-causal self-attention with per-token-type expert weights,
flow-matching action chunks — see the pi0 log entry above), a distinct
question surfaced that the existing single-vs-mixture-of-Betas gate
(`docs/milestones/M4.md` Sec. 4.2/8, not yet run) does **not** cover.

Classical acquisition optimization (EI/UCB/PI on a GP) is always two
separate steps: (1) a cheap pointwise posterior query `(μ(x), σ(x))`, then
(2) an *outer* search (multistart L-BFGS/CMA-ES/DIRECT) over that surface,
because acquisition surfaces are routinely multimodal even when the
posterior itself is well-behaved. `ActionHead` currently collapses both
steps into one: a single forward pass through a fixed-depth stack, reading
out one token's final state, straight to a `Beta(α,β)` per dimension. There
is no explicit search step anywhere.

This is a **different axis** from the multimodality gate. That gate asks
"is the oracle's target distribution over `x*` unimodal or multimodal" (a
question about the *shape* of the answer). This one asks "even for a single
well-defined target, can one feedforward pass locate the argmax of an
implicitly-represented, possibly-multimodal surface at all, or does
argmax-finding structurally require iteration regardless of how many modes
the final answer has" (a question about *how much computation* finding the
answer takes). Conflating the two would hide a real architectural gap.

## What we considered

Three design directions, none implemented yet:

### Option A — multi-start via parallel candidate tokens

Generalize `action_query` (`action_head.py:162,213`) from 1 token to `K`
tokens. All `K` self-attend among themselves and the 4 aux tokens (already
supported — `self_mask` at `action_head.py:217` is already an all-ones
`[T,T]` mask, agnostic to `T`), each cross-attends per-layer into the same
cached `hidden_states[i]` (one PFN forward pass regardless of `K`, since
train-token self-attention doesn't depend on how many query/test tokens
read it — see `models/pfn.py`'s train/test split). At readout, run every
candidate token through both `policy_head` and `value_head`, and commit to
the argmax-by-value candidate.

Motivation: `value_head` already exists (added for EXIT's branch valuation,
`aa7068f`/`f8e3e97`) and is exactly the scoring function a multi-start
search needs to pick a winner — this reframes an existing component rather
than adding a new one. Trainable with a best-of-K / winner-take-all loss
(only the closest-to-oracle candidate gets the imitation gradient each
step, à la Multiple Choice Learning / diverse beam search), which has a
notable side effect: **best-of-K training simultaneously answers the
still-open multimodality question** (if the oracle target is unimodal, the
K candidates just learn to agree; if multimodal, they specialize onto
different modes) **and** gives emulated multi-start search — one mechanism,
two currently-separate open questions.

### Option B — recursive/looped refinement over the same cached context

Loop `ActionHead`'s own block stack (self-attn → cross-attn into cached
`hidden_states` → FFN) multiple times before reading out, either with
shared weights (recurrent/"Universal Transformer"-style, zero extra
params) or a small number of distinct unrolled stages. Unlike pi0.5's
iterative loop — which exists to amortize repeated *expensive* VLM-prefix
re-encoding across 10 flow-matching denoising steps, hence their whole
block-causal-mask/KV-cache design (Appendix B/D, `arXiv:2504.16054`) — the
PFN forward pass here is already computed exactly once and cached
regardless, so looping just `ActionHead`'s own cheap stack against that
fixed cache costs almost nothing. Closer in spirit to unrolled gradient
ascent on an implicit surface than to denoising. pi0.5's own authors use
this same framing for their two-stage (subtask → action) inference: *"a
recipe that more closely resembles chain-of-thought... or test-time
compute methods"* (`arXiv:2504.16054`, Sec. III-A). Composes with Option A
(K candidates, each refined for a few iterations, then value-selected).

### Option C — full flow matching in x-space (pi0-style), deferred

Treat the query point itself as the thing being iteratively denoised:
`x^0 ~ N(0,I)`, integrate `x^{τ+δ} = x^τ + δ·v_θ(x^τ, context)` toward the
oracle's `x*`, trained with the same straight-line flow-matching loss pi0
uses (`arXiv:2410.24164` Sec. IV). Doesn't require a multimodal target —
collapses to learning a straight vector field toward one point for a
deterministic oracle — so it isn't blocked by the multimodality gate
either. Real added machinery (noise schedule, integration-step count, a
new loss shape) for a benefit that's currently speculative; consistent
with this project's pattern of gating additions behind evidence (same
reasoning already applied to the mixture-head decision), this is the
fallback if A/B prove insufficient, not the first thing to build.

## What we learned

The single-vs-mixture question (shape of the target) and the
single-shot-vs-iterative question (depth of search needed to find the
target) are independent axes that the current `ActionHead` docstring only
addresses the first of. Best-of-K training over parallel candidate tokens
(Option A) is a rare case where one mechanism change answers both at once,
and it reframes rather than adds to the architecture (`value_head` already
exists for a different reason and slots directly into the role of
candidate-selector). Recursive refinement (Option B) is cheap specifically
*because* the PFN backbone is frozen and called once — a benefit pi0.5
doesn't get, since their backbone re-encoding is the expensive part their
own caching scheme is built to avoid.

## Status / next steps

Parked as design options, not yet implemented or prioritized against each
other. Planned order if/when pursued: A first (cheapest, reuses
`value_head`, no new loss machinery beyond best-of-K selection), B as a
near-free addition on top, C held in reserve. A follow-up diagnostic —
training PFN+ActionHead to find the argmax of a *known, cheap, closed-form*
acquisition function (rather than the full privileged-search oracle) to
isolate whether the architecture can do argmax-finding at all, independent
of whether the acquisition function itself is being learned correctly at
the same time — was proposed in the same conversation and is being scoped
separately (see the next log entry once it exists, or `docs/ROADMAP.md` if
adopted into the phased plan).
