# M2: PFN transformer + bar distribution, trained checkpoints, prior/data/PFN overlay notebook

**Date:** 2026-08-28
**Related:** `docs/milestones/M2.md`, `docs/OPEN_QUESTIONS.md` #8,
`docs/log/2026-08-27-pfns4bo-bnn-prior-comparison.md`

## Motivation / hypothesis

M2 (frozen PFN checkpoint) was next per the roadmap, depending on M1's
`BNNPrior`. Before implementing, checked PFNs4BO's and ifBO's actual PFN
transformer and training procedure (`pfns4bo/transformer.py`,
`pfns4bo/train.py`) for the reference attention-masking mechanism and
training-loop details, rather than re-deriving them from the design doc
alone — the user flagged their code as messy going in, so the goal was
extracting the essential facts, not porting their code wholesale.

## What we tried

**Architecture confirmation.** PFNs4BO's `TransformerModel.generate_D_q_matrix`
builds a single attention mask fed into one homogeneous transformer stack
(not separate self-/cross-attention modules): train columns open to every
row (train-train bidirectional + test-train cross-attn), test columns
closed to everyone except each token's own diagonal entry. Confirms the
already-tested prototype's approach
(`archive/src/exit/claude/pfn-explore-exploit-repo/repo/src/model/pfn.py`,
a single masked encoder stack too) is the right shape, with one deliberate
difference: that prototype (and our port) doesn't include the diagonal
self-attention term for test tokens, matching the design doc's "never to
themselves" more literally — PFNs4BO's own version most likely includes it
just to guard against an all-masked-row edge case in their more general
code, not as a design principle. No positional encoding either — confirmed
absent from PFNs4BO's config paths that matter.

**Training-loop confirmation.** `pfns4bo/train.py`: AdamW, cosine schedule
with warmup, and critically — `single_eval_pos` (the train/test split
point) is *resampled every step*, not fixed, so the model learns to handle
arbitrary context sizes rather than one fixed size. NLL loss via the
`BarDistribution`, computed only on the test/query portion of the sequence.

**Implementation** (all in `src/anytimeacquisition/models/` and
`src/anytimeacquisition/pipelines/`):
- `models/pfn.py`: ported the tested prototype's `PFN` near-verbatim.
- `models/bar_distribution.py`: ported PFNs4BO's `BarDistribution` (NLL,
  mean, median, variance, mode), trimmed of their smoothing/mean-prediction-
  loss/EI/PI/UCB machinery (EI/PI/UCB belongs to M6's baselines, not the
  PFN's own output head), fixed `[0,1]` borders (bounded — see
  `docs/OPEN_QUESTIONS.md` #8, resolved this session). Added `entropy()`,
  which the original doesn't have, closed-form per the design doc.
- `pipelines/train_pfn.py`: `train_pfn()` (fresh `BNNPrior` instance per
  step, randomized train/test split size per step, AdamW + cosine warmup,
  checkpoint save) and `load_pfn_checkpoint()`. Also
  `plot_prior_data_and_pfn_1d()`: the requested diagnostic — true function +
  train data + the PFN's predictive density (`softmax(logits)/bucket_width`)
  as a heatmap, all on one plot.
- `notebooks/pfn_prior_overlay_showcase.ipynb`: self-contained (trains a
  small `x_dim=1` PFN inline rather than depending on a gitignored
  checkpoint file), architecture sanity checks, several overlay plots
  across seeds/context sizes, and the entropy-vs-context-size check M2's
  checklist asked for.

## Result

Tests (`tests/test_pfn.py`, `tests/test_bar_distribution.py`,
`tests/test_train_pfn.py`, 15 new, all passing): permutation invariance
over the train set, no test-test leakage (both a forward-pass equality
check and a stronger direct gradient-isolation check —
`d(logits at test point i)/d(x at test point j != i)` is exactly zero, not
just numerically close), bar-distribution entropy matches the analytic
`U(0,1)` case exactly, NLL decreases over a short training smoke run,
checkpoint save/load round-trips.

Two real training runs (not just smoke tests): `x_dim=2`, `d_model=64`, 3
layers, 500 steps — train NLL `+0.14 -> -0.84`, eval MSE `0.093 -> 0.031`
over ~150s on CPU (checkpoint saved, gitignored, regenerable via
`pipelines/train_pfn.py`'s `__main__`). A second, smaller `x_dim=1` run
(300 steps, feeds the notebook) converged further: train NLL
`+0.17 -> -0.99`, eval MSE `0.10 -> 0.011`. The overlay plots show real
in-context learning — predictive density visibly tracks the true function
even at `n_train=3`, and visibly tightens around it by `n_train=20`.

**Entropy-vs-context-size is not monotonic**, confirming the concern
M2's checklist carried over from the original prototype's `x_dim=2` run:
mean predictive entropy over a fixed grid at `n_train = 2, 5, 10, 20` was
`-0.619, -0.782, -0.611, -0.645` — drops from 2->5, then rises back up at
10 before flattening. Same qualitative finding as before, now reproduced
at `x_dim=1` with the current (much-changed) prior and a real (not smoke)
training run — this isn't an artifact of the earlier flat-draws bug or an
undertrained checkpoint, it's a more persistent property worth taking
seriously before the explore branch (M5) leans on this entropy signal.

## What we learned

Extracting "the essential mechanism" from PFNs4BO's reference code (a
single mask into one homogeneous encoder stack, rather than assuming
separate self-/cross-attention modules were needed) was more useful than
their code's own generality — most of `transformer.py`'s surface area
(global attention tokens, style encoders, multiple decoder heads, per-layer
different init) exists for capabilities this project doesn't need yet. The
already-tested prototype's simpler, purpose-built implementation was the
right thing to port, with the reference code serving to *validate* that
approach's masking pattern rather than as something to copy from directly
— consistent with the user's "notice their code is messy" framing.

Also: a strong, forward-pass-level "no leakage" check (comparing full
output tensors before/after perturbing one test point) is good, but a
*gradient*-level check (is `d(other output)/d(this input)` exactly zero)
is a strictly stronger claim about the actual computational graph, cheap to
add, and worth making the default pattern for this kind of invariance test
going forward.

## Status / next steps

Adopted. `docs/OPEN_QUESTIONS.md` #8 resolved (bounded bar distribution).
M2's checklist (`docs/milestones/M2.md`) updated to reflect what's actually
built, including the deviation from the original full-support-bar-
distribution plan. M4 (ActionHead) is next per the roadmap, depending on
this frozen-PFN interface (`PFN.forward(..., return_hidden=True)` already
exposes the per-layer hidden states M4's cross-attention needs).
