# Most BNNPrior draws were flat — independent init_std/width sampling, fixed via crit-scaling

**Date:** 2026-08-27
**Related:** `docs/milestones/M1.md`, `src/anytimeacquisition/priors/bnn.py`

## Motivation / hypothesis

While demoing M1's 1D/2D environment plots, most sampled BNN "true
functions" looked nearly flat or very simple — only an occasional draw
showed real structure. Before treating that as expected behavior (the
design doc already flags "achievable range varies wildly by instance" as
known), wanted to check whether it was *incidentally* this skewed, or a
structural artifact of how the prior samples architectures.

## What we tried

Sampled 300 fresh instances (`x_dim=1`) and computed, per instance, the
standard mean-field/Xavier-style per-layer variance-scaling quantity
`crit = init_std**2 * width` (≈1 preserves variance layer-to-layer; well
below it, signal vanishes toward a near-constant output; well above it, the
network enters the chaotic/rich regime). Correlated `crit` (and `depth`)
against a complexity proxy (`raw_output.std()` over a grid), then
visually swept `crit ∈ {1,2,4,8,16}` × `depth ∈ {3,6,10,16}` with
controlled weights to see actual curve shapes at each point, not just a
correlation coefficient.

Root cause found: `init_std` was sampled uniformly from a fixed range
*independent of the width actually drawn for that instance* — not tied
together as `init_std ∝ 1/√width` (the standard scaling). So `crit` ended
up scattered ~0.25–4.7 across draws almost by chance, with most landing at
or below the "preserves variance" point.

## Result

- `corr(complexity, crit) = 0.68` (real, not noise); `corr(complexity,
  depth) = 0.16` (depth mattered far less than the crit mismatch).
- Concrete instances at the old crit distribution: `crit=0.48` → raw output
  range **0.006** over the whole domain (flat at float-noise scale);
  `crit=1.48` → range **0.22** (a near-straight ramp); `crit=3.89` → range
  **~3.5** (genuine multi-bump structure, the rare "interesting" draw).
- The crit×depth sweep showed a depth-dependent "interesting but not pure
  noise" band roughly in `crit ∈ [2, 8]` across `depth ∈ [3, 16]` — below it,
  draws are boring regardless of depth; above it (particularly `crit=16` at
  `depth≥10`), curves degrade into high-frequency, noise-like jaggedness.
- **Fix**: sample `crit` directly (log-uniformly, since complexity tracks
  `log(crit)` more than `crit` itself) from a `crit_range` parameter
  (default `(2.0, 8.0)`, picked from the sweep), then derive
  `init_std = sqrt(crit / width)` using the actually-sampled width —
  replacing independently-sampled `init_std_range`.
- Re-ran the same 300-instance check after the fix: median raw-output std
  0.66 → (was ~0 for most); median ECDF-normalized y-range per instance
  0.34 (was dominated by <0.05-wide bands); 54% of instances now span
  >0.3 of the unit interval, 35% span >0.5. 1D/2D demo plots re-generated
  and visually confirm real gradients/structure in most draws now, not just
  the occasional lucky one.

## What we learned

This is a general lesson about randomly-initialized tanh (or any bounded-
activation) MLP families used as synthetic BO test functions, not specific
to a bug in this one implementation: **sampling architecture (width, depth)
and init scale independently, even from "reasonable-looking" ranges, does
not give you a uniform-ish spread of function complexity** — it gives you
mostly-degenerate (flat or saturated) draws with occasional rich ones,
because complexity is governed by a *derived, multiplicative* quantity
(`init_std² × width`, compounded over depth) that a fixed init_std range
doesn't control for at all. If a future prior/environment ever adds a
similar per-instance random architecture again (deeper MLPs, different
activations, etc.), sample the mean-field scaling quantity directly and
derive the raw hyperparameters from it, not the other way around.

Also worth remembering procedurally: this was found by actually plotting
draws and looking, not by reading the sampling code and reasoning about it
in the abstract — the bug was in a part of the code (`reset()`) that had
already been ported, tested (output range, differentiability, batch
independence), and marked done in M1. Tests for "does it run and produce
valid-shaped output in `[0,1]`" don't catch "is the *distribution* of what
it produces actually useful."

## Status / next steps

Adopted — `crit_range` replaces `init_std_range` in `BNNPrior` and
`configs/priors/bnn.yaml`. `docs/milestones/M1.md` updated to point here.
`crit_range=(2.0, 8.0)` is a reasonable default, not a tuned optimum — if
later work (M2 PFN training, M5 search) wants an even richer or tamer
family, it's one config override away, informed by the sweep numbers above
rather than needing to re-run the investigation from scratch.
