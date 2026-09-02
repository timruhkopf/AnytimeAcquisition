# EI-argmax isolation diagnostic: can ActionHead find the argmax of a known acquisition function, decoupled from learning one

**Date:** 2026-09-02
**Related:** `docs/log/2026-09-02-actionhead-search-depth-design-options.md`
(the K-candidate-token / recursive-refinement / flow-matching options this
diagnostic exists to gate), `docs/milestones/M5.md`'s still-open "1D
interpretability diagnostic" checklist item (narrower overlap — EI +
argmax only, not entropy/UCB/PI/policy-density),
`src/anytimeacquisition/models/baselines/pfn_acquisition.py`,
`src/anytimeacquisition/pipelines/action_head_ei_diagnostic.py`

## Motivation / hypothesis

**The problem formulation, stated precisely, because it's easy to
conflate with the full EXIT task:** this repo's actual, eventual ask of
`ActionHead` is harder than plain argmax-finding. It must simultaneously
(a) implicitly represent *what a good acquisition function even is* — there
is no closed-form target it's told to maximize; it has to infer this from
privileged-search-derived training signal (the exploit/explore oracles) —
**and** (b) return, in one forward pass, the point that maximizes it. The
current EXIT training signal entangles both: if a trained `ActionHead`
underperforms, there is no way to tell whether the failure is in (a) the
oracle/target itself being poorly defined, noisy, or not genuinely reduce
regret (already a live concern — see
`docs/log/2026-08-28-explore-search-input-optimization-and-teacher-forcing.md`'s
finding that only 18.5% of explore corrections strictly reduce regret), or
in (b) the architecture's single-shot cross-attention readout being
structurally unable to locate an argmax at all, even when the surface it
should be maximizing is completely well-defined.

**Hypothesis to isolate:** hold (a) fixed and known — give `ActionHead` a
genuine, closed-form, unambiguous acquisition function (Expected
Improvement, computed directly off the *same* frozen PFN's own PPD it
already cross-attends into, not a separate model) as the sole training
target, and test only (b). If the architecture can't learn this simpler,
fully-specified task, it is very unlikely to succeed at the harder joint
task (define + maximize) the real EXIT pipeline asks of it. Same logic a
T-maze uses to isolate long-horizon credit assignment from the rest of
RL's complexity before trusting a result on a full environment, applied
here to argmax-finding instead of credit assignment.

## What we tried (design + smoke test only — see Result)

- `models/baselines/pfn_acquisition.py::expected_improvement` — closed-form
  EI over the frozen PFN's own bar-distribution PPD. Explicit user
  direction: **do not hand-derive this formula** — port/adapt it from a
  real reference implementation instead. Found that PFNs4BO's actual
  `ei()` (maximize convention) is already vendored in this repo at
  `archive/src/utils/bar_distribution.py` (lines 135–149); the current
  `models/bar_distribution.py`'s own docstring already documents that this
  machinery was deliberately dropped during that port and "belongs to
  M6's classical baselines instead." Adapted (not re-derived) the same
  per-bucket clamping trick, algebraically mirrored for this project's
  minimize convention, and — more importantly than the algebra — cross-
  checked numerically against a Monte Carlo estimate in
  `tests/test_pfn_acquisition.py` (sampling directly from the piecewise-
  uniform density, not through the PFN).
- `pfn_ei_argmax` — the argmax-finding oracle itself. Explicit user
  direction: **do not reuse `search/explore.py`'s multistart-GD pattern**
  — that optimizer can itself get stuck in local optima, which would make
  the oracle a second, entangled error source on top of whatever this
  diagnostic is trying to isolate. Replaced with something simpler and
  more trustworthy: a dense `torch.linspace` grid (x_dim=1 only — a third
  user-directed simplification, both for tractable dense coverage and for
  direct visualization), one batched PFN forward call (train-side
  self-attention runs once regardless of grid density —
  `models/pfn.py`'s test tokens never influence train tokens), argmax over
  the grid. No optimizer, no local-minima risk — the only approximation is
  grid resolution, directly checkable (`n_grid` higher vs. lower), not an
  optimizer-convergence question.
- `pipelines/action_head_ei_diagnostic.py` — a fresh, standalone pipeline
  (explicit user direction: do **not** extend the structurally similar but
  currently orphaned `pipelines/action_head_posterior_distill.py`, whose
  Hydra config was manually deleted before this session). Reuses that
  file's proven three-stage failure-isolating *methodology* (not its code):
  memorize one fixed context → generalize across fresh contexts with
  held-out evaluation → blind ablation (PFN hidden states zeroed) — now
  against the EI-argmax target instead of a pure posterior-mean-argmin
  one. EI is a strictly harder target than that precedent's: it depends on
  the posterior mean *and* variance *and* the current incumbent, not just
  "where is the mean lowest." `plot_ei_diagnostic` overlays the
  ground-truth EI curve, its argmax, and the trained `ActionHead`'s own
  `beta_mode` prediction per held-out context, logged to MLflow
  (`mlflow.log_figure`) every run.

## Result

**Real-scale run on `ulysses` (x_dim=1, `pfn_smoke_xdim1.pt`,
`memorize_steps=300 generalize_steps=2000 eval_contexts=100 n_grid=1000`,
`priors.batch_size=16`, `ActionHead` defaults):**

```
memorize final loss:            5.4987 -> 0.2318
generalize (real)  held-out L1: 0.1745
generalize (blind) held-out L1: 0.2955
PASS -- real clearly beats blind  (0.1745 < 0.8 * 0.2955 = 0.2364)
```

Re-run once more after fixing two batch-dimension bugs surfaced while
adding the PFN-posterior plot panel (see the second commit on
`ei-argmax-diagnostic`) — identical numbers both times (same seed, the fix
was plotting-only), which is itself a useful consistency check that the
fix didn't perturb training.

`plot_ei_diagnostic`'s overlay (4 held-out contexts, PFN posterior
mean±std on top, EI curve + true argmax (red) + `ActionHead`'s own
`beta_mode` prediction (blue) on the bottom) shows a real but imperfect
correspondence: context 1 is a close match, contexts 0/2/3 show the
policy landing in the right general region but off the true argmax by a
non-trivial margin — consistent with the aggregate 0.17 mean L1 on a
`[0,1]` domain, not a coincidence of the aggregate number.

## What we learned

**The architecture passes this isolated test at real scale.** The
frozen-PFN cross-attention pathway clearly carries information the blind
ablation cannot access (0.1745 vs. 0.2955 held-out L1 — not a marginal
gap), and the memorize stage confirms the Beta head can fit a single known
EI-argmax target from a cold start (5.50 → 0.23). This is real evidence
that `ActionHead`'s single-shot, single-query-token readout *can* do
argmax-finding on a well-defined, closed-form, unimodal-ish 1D acquisition
surface — the narrower of the two questions this diagnostic set out to
separate (see Motivation) — without needing the search-depth options
(K-candidate tokens / recursive refinement / flow matching) from
`docs/log/2026-09-02-actionhead-search-depth-design-options.md`, at least
not at this scale/dimensionality.

**What this does NOT establish, stated plainly so it doesn't get
overclaimed later:** (1) a 0.17 mean L1 on `[0,1]` is a real gap, not
convergence to the true argmax — the overlay plot shows genuine misses,
not just noise around a tight fit; (2) x_dim=1 only, on one smoke-scale
PFN checkpoint — the multimodal, higher-x_dim, or harder-EI-landscape case
(e.g. context 0's sharp EI ridge) is untested; (3) this says nothing yet
about the *other*, harder, entangled question (can `ActionHead` also learn
what a good acquisition function is, not just maximize a given one) —
that's still the full EXIT pipeline's open problem, unaffected by this
result either way.

## Status / next steps

**Adopted as a working diagnostic; first real-scale result in hand.**
Passes at x_dim=1 on the smoke checkpoint. Next candidates, not yet
started: (a) run against `pfn_ulysses_real.pt` or another non-smoke
checkpoint once one exists at x_dim=1, to check the result isn't an
artifact of the smoke PFN's own (possibly under-calibrated) posterior;
(b) push `n_grid`/`generalize_steps` further to see whether the 0.17 L1
floor is an optimization/data-budget limit or closer to the architecture's
ceiling; (c) if a higher-x_dim checkpoint becomes available, this
diagnostic's grid-based oracle stops being cheap/exhaustive (per
`pfn_ei_argmax`'s own docstring) — would need revisiting before extending
past x_dim=1. Given the pass here, the search-depth options in the sibling
log entry are not urgently motivated by this result alone; revisit them if
(a)/(b) surface a real ceiling rather than a budget limit.
