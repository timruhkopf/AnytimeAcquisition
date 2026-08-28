# Why pi0/pi0.5 use separate expert weights + a fine-tuned backbone, and why the ActionHead deliberately differs (frozen PFN)

**Date:** 2026-08-28
**Related:** `docs/milestones/M4.md`, `docs/ROADMAP.md` Phase 4 and "Design
reference" section, `src/anytimeacquisition/models/action_head.py` (module
docstring cites the same papers, more tersely)

## Motivation / hypothesis

M4's ActionHead is explicitly modeled on pi0/pi0.5's action-expert design
(`docs/ROADMAP.md`: "pi0.5-style VLA integration"). Before implementing it,
the ask was to check the actual papers for best practices rather than work
from the ROADMAP's paraphrase alone, and to reason explicitly about which
parts of their design we should copy vs. deliberately diverge from, rather
than importing the pattern uncritically because it's the cited inspiration.

Two specific things needed checking: why they use separate ("mixture of
experts"-style) weights for action/state tokens instead of routing them
through the same weights as the VLM tokens, and why their VLM backbone is
fine-tuned rather than frozen — since our ActionHead does the opposite
(frozen PFN) and that divergence needed to be a *reasoned* choice, not just
inherited from `docs/ROADMAP.md`'s existing phrasing.

## What we tried

Fetched and read the primary sources directly rather than relying on
secondary summaries: pi0 (Black et al., "$\pi_0$: A Vision-Language-Action
Flow Model for General Robot Control", arXiv:2410.24164, Sections III–IV)
and pi0.5 (Black et al., "$\pi_{0.5}$: a Vision-Language-Action Model with
Open-World Generalization", Physical Intelligence, pi.website/blog/pi05,
read as a PDF, Sections III–IV / Fig. 3).

Specifically searched for the papers' own *stated rationale* for (1) the
separate-expert-weights choice and (2) fine-tuning vs. freezing the
backbone — not just the mechanism — to avoid inventing a rationale and
attributing it to the paper.

## Result

**The papers don't give a first-principles rationale for either choice.**
This is worth stating plainly rather than glossing over: it would have been
easy to write a docstring that implies pi0 "explains why" MoE-style weights
or fine-tuning work, and that would misrepresent the source.

- **Separate expert weights**: pi0's only stated justification is
  empirical — "Building on Transfusion, we additionally found that using a
  separate set of weights for the robotics-specific (action and state)
  tokens led to an improvement in performance." (Sec. IV). Built on a prior
  technique (Transfusion), not derived from first principles in-paper.
- **Fine-tuned, not frozen, backbone**: no rationale is stated at all. The
  papers describe initializing from a pretrained VLM (PaliGemma) and then
  training all weights through both pre- and post-training — never argue
  for why over freezing.

What follows is *inferred* reasoning (ML fundamentals, not paper quotes),
clearly separated from the above:

**Why separate expert weights probably helps them:** two very different
token populations share one sequence — image/language tokens (large
pretrained vocabulary, discrete-ish semantics) and continuous robot
state/action tokens (small, dense, numeric). Forcing both through identical
Q/K/V/FFN weights creates a real capacity/specialization tension; MoE-style
routing by token type (not learned gating) is a standard fix — shared
attention operation for cross-stream information flow, separate weights per
stream for modality-appropriate processing. There's also a concrete
inference-cost payoff specific to their setup: the action expert (300M) is
much smaller than the VLM (3B); at inference the VLM's KV cache is computed
once while only the small expert re-runs through 10 flow-matching denoising
steps — that asymmetric compute split is what gets them to real-time
control frequencies (up to 50Hz), and it requires separate weights to be
possible at all.

**Why fine-tune rather than freeze:** pi0.5's entire contribution is
*transfer* — bridging a large domain gap between broad web-scale semantic
pretraining (captioning, VQA, object localization) and a narrow, precise
downstream skill (continuous low-level robot control in previously-unseen
homes). A frozen VLM's representation was optimized for "what's in this
image," not "what's the precise gripper trajectory to close this cabinet" —
there's a real gap to bridge, and the paper's own abstract centers on how
joint co-training across heterogeneous sources (web data, cross-embodiment
robot data, high-level subtask labels) is what enables broad generalization
— mechanically impossible if the backbone can't update. They also don't
rely on freezing to prevent catastrophic forgetting; they solve that by
co-training on the original web-scale data alongside robot data, not by
locking weights.

**Applying this to our case — the question that actually matters:** *is
there a domain gap between what the PFN was trained to do and what the
ActionHead needs from it?*

For pi0.5: yes, a large one. For us: **no.** The PFN (M2) was purpose-built
from scratch to produce exactly the representation the ActionHead consumes
— a permutation-invariant summary of observed `(x,y)` pairs yielding a
calibrated posterior predictive distribution. There's no broader pretrained
knowledge being narrowed down; the PFN's entire existence is already scoped
to this one task, so the transfer-learning argument for fine-tuning doesn't
transplant here.

Freezing has a real, pipeline-specific cost that's worth naming precisely
rather than treating as a generic caution: M5's explore branch does
gradient descent directly on "the frozen PFN's closed-form predictive
entropy" to find points that reduce genuine epistemic uncertainty
(`docs/ROADMAP.md` Goal, point 2). That only works if the entropy surface
is a trusted, fixed oracle. If ActionHead training pressure could reshape
the PFN, the policy could learn to reshape the very entropy landscape it's
scored against — a reward-hacking-shaped failure mode, not a calibration
improvement. It would also compound across EXIT's iterative retraining
rounds: the ROADMAP already flags that raw `(context, x*, y*)` tuples go
stale the moment the ActionHead retrains (Phase 5); a drifting PFN would
make *every* prior round's oracle labels stale too, not just the KV cache.

Scale mismatch matters too: pi0.5 has hundreds of hours of diverse data
plus a huge web co-training mixture specifically to prevent a fine-tuned
backbone from collapsing/overfitting. We have a small transformer and a
comparatively narrow training signal (one synthetic BNN-prior family) —
fine-tuning here would carry real overfitting/collapse risk with none of
their anti-forgetting machinery to counteract it.

On separate expert weights specifically: for us this isn't really an
optional design choice inspired by pi0 — it's a *mechanical consequence* of
freezing. Frozen weights can't be trained, so the ActionHead necessarily
needs its own parameters; there's no "share weights with the PFN" option
once freezing is decided. Multi-layer tapping *is* a genuine independent
choice, for a different reason than pi0's (theirs is free — every layer's
joint attention happens anyway as part of one forward pass; ours costs
extra parameters and compute deliberately spent) — the PFN's earlier layers
likely carry more raw, less-abstracted information about individual
observations than the final layer (optimized for NLL, not for preserving
everything useful to an acquisition policy), so tapping only the last layer
risks losing signal the final layer had no incentive to keep.

## What we learned

Freeze is the right call for the PFN, for reasons that mirror pi0.5's logic
in *structure* ("does this backbone need adaptation to close a domain gap,
or is its frozen behavior the valuable asset?") without importing their
actual conclusion — our answer to that question is the opposite of theirs,
for identifiable reasons, not by default.

Equally important: don't cite a paper's mechanism as if it were also the
paper's stated justification. Checking primary sources directly surfaced
that pi0's own justification for separate expert weights is a one-line
empirical result, not an argued design principle — worth knowing when
deciding how much weight to put on "pi0 does X" as evidence for doing X
ourselves elsewhere in this project.

Freezing does cap the ActionHead's ceiling — worth stating honestly rather
than treating freezing as a free lunch. If the PFN's frozen representation
is missing information genuinely useful for acquisition (not just
calibrated uncertainty but some finer structure), fine-tuning could in
principle recover it — exactly the tradeoff the frozen-backbone-probing
line of VLA research studies directly (surfaced during the same search).
M2 already flagged a concrete, unresolved concern in this direction:
predictive entropy doesn't shrink monotonically with context size — an
open question for whether the frozen PFN's entropy surface is fully
trustworthy as the explore branch's oracle.

## Status / next steps

Adopted — freeze the PFN, as already implemented in
`src/anytimeacquisition/models/action_head.py` and now justified here in
more depth than the module docstring carries. Not treated as beyond
question: if M5's explore branch underperforms and the cause traces back to
the PFN's frozen representation lacking acquisition-relevant signal (see
the non-monotonic-entropy finding, `docs/log/2026-08-28-m2-pfn-and-bar-distribution.md`),
revisit fine-tuning then, with the cost above (oracle staleness across EXIT
rounds, reward-hacking risk on the entropy surface) explicitly weighed
against whatever the frozen representation turns out to be missing — not
before there's concrete evidence the ceiling is actually being hit.
