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

**Not yet available.** Only a tiny (20-step) local CPU smoke test has been
run so far — confirms the pipeline runs end to end, logs to MLflow
correctly, no shape/correctness issues (11/11 unit tests pass, including
the Monte Carlo cross-check of the EI formula and a check that the
EI-argmax target genuinely differs from the easier posterior-mean-argmin
one). That smoke run's own verdict (`INCONCLUSIVE/FAIL`, `real_eval_l1`
barely below `blind_eval_l1`) is **not evidence of anything** — 20 steps
is far too few to expect convergence either way; it is not a real-scale
result. A real-scale run is being dispatched to `ulysses`
(see this repo's `CLAUDE.md` for what that machine is) as of this entry.

This section, and "What we learned" below, get filled in once that real
run's `real_eval_l1`/`blind_eval_l1`/pass-fail numbers exist, or once the
`ActionHead` architecture has been iterated on (per
`docs/log/2026-09-02-actionhead-search-depth-design-options.md`'s
K-candidate-token / recursive-refinement / flow-matching options) to
actually meet what this experiment demands — appended, per this log's own
append-only convention, not rewritten in place.

## What we learned

Pending — see Result.

## Status / next steps

Design complete, unit-tested, smoke-tested locally (pipeline runs
end-to-end, no real-scale signal yet). Next: a real-scale run on
`ulysses`, then return to this entry (or a cross-linked follow-up) with
the actual numbers and verdict. If the architecture fails even this
simpler, fully-specified target at real scale, that is meaningful evidence
for pursuing the search-depth options (K-candidate tokens / recursive
refinement / flow matching) before spending more effort on the harder,
entangled full-EXIT privileged-search target.
