# Milestones

Current status and priorities — see `docs/ROADMAP.md` for *why* the system
is shaped this way. This file answers *where it actually stands*, split the
way debugging it requires: component-level status (does each piece work in
isolation), conceptual status (does the whole idea work), named bottlenecks,
and a prioritized experiment order. Not a linear phase checklist — several
of these are independently gate-able and don't need to happen in numeric
order except where noted.

## Component-level status

Each of these has been checked on its own, independent of whether the full
pipeline works end to end:

- **BNN prior** (`priors/bnn.py`) — works. Crit-scaling fix resolved an
  earlier flat-draw issue (init scale needs to compound with depth, not be
  sampled independently of it); aligned with PFNs4BO/ifBO conventions
  (noise, input scaling, sparseness, spurious dims).
- **PFN + bar distribution** (`models/pfn.py`, `models/bar_distribution.py`,
  `trainer/pfn_trainer.py`) — works. Trains; checkpoints exist at smoke
  scale (`pfn_smoke_xdim1.pt`) and variable-x_dim scale
  (`pfn_variable_xdim_smoke.pt`, trained with `variable_dim_min` so a single
  checkpoint handles `x_dim` 1 through 6).
- **Exploit search** (`search/exploit.py`) — works. Validated against a
  dense-grid ground truth (~0.004 mean gap on a 2D instance, not just
  shape-checked); the never-worse-than-the-known-incumbent guarantee holds
  (Adam's steps aren't monotonic, so this needed an explicit fallback, and
  it does what it's supposed to). **Not built**: the multi-basin
  exploitation-pull fix — it keeps only its single best restart, discarding
  the rest, so a policy trained on its output only ever sees one basin per
  correction even when several exist.
- **Explore search** (`search/explore.py`) — mechanically sound (teacher-
  forcing is real, not post-hoc y-overwriting; no cross-instance leakage;
  best-so-far tracking around Adam's non-monotonicity), but **label quality
  is checkpoint- and scale-dependent, not a fixed property of the
  mechanism**. Measured directly (`greedy_regret`,
  `pipelines/explore_search_playground.py`):

  | checkpoint | x_dim | corrections | frac. regret improved | frac. regret worsened | proxy↔regret correlation |
  |---|---|---|---|---|---|
  | `pfn_smoke_xdim1.pt` | 1 | 314 | 18.5% | 14.3% | 0.86 |
  | `pfn_variable_xdim_smoke.pt` | 6 | 1429 | 55.8% | 20.6% | 0.36 |

  Both checkpoints are smoke-scale, not converged — these numbers are a
  reason to keep measuring on whatever checkpoint is actually in use, not a
  settled property of the search itself. The correlation dropping while the
  improved-fraction rises is worth sitting with, not averaging away: a
  weaker proxy relationship *and* a better hit rate happened together, at
  higher dimensionality, on a different checkpoint — more than one variable
  changed, so neither number alone should be trusted as "the" explore-search
  quality figure. **Important correction to how these numbers must be read:**
  both were measured with `build_explore_buffer`'s `require_improvement=True`
  gate already active (its default) — i.e. every correction counted above
  had *already* passed a cheap proxy filter (its own weighted-NLL score
  improved) before the regret check ran. That gate existing was previously
  understated here as "no filtering exists" — wrong; the gate exists, and
  these numbers are direct evidence it is *insufficient on its own*: even
  restricted to proxy-filtered corrections, a large fraction still don't
  reduce real regret. See Bottleneck 1 below for the actual fix.
- **ActionHead single-shot argmax-finding**
  (`pipelines/action_head_ei_diagnostic.py`) — works, confirmed twice (a
  smoke checkpoint and a genuinely trained non-smoke checkpoint, same
  qualitative result both times): held-out `|beta_mode - true EI argmax|` is
  clearly lower with the real cross-attention link than with it blinded
  (PFN hidden states zeroed) — `0.17`–`0.20` real vs. `0.25`–`0.31` blind,
  well past the 20%-margin pass bar. This isolates and confirms *only* that
  the architecture can find the argmax of an already-known, closed-form
  target (Expected Improvement, computed directly off the PFN's own
  posterior) — it says nothing about whether the actual training targets
  (exploit/explore corrections) are themselves good targets to imitate.
- **k-step privileged planning** — promoted to production
  (`search/kstep_explore.py`, tested,
  `notebooks/kstep_explore_search_labeling.ipynb` kept for the worked
  demonstration/plots it was promoted from). The mechanism works (one PFN
  call per gradient step scoring a jointly-optimized short plan, not one
  call per planned point — this is what keeps it cheap), and the
  attribution problem described in `docs/ROADMAP.md` (a joint plan's score
  isn't any one point's own value) is not just a theoretical concern:
  measured directly, the joint score overstated a selected point's real,
  independently-rescored contribution by as much as `17.1` in the prototype
  (weighted-NLL units), and, in a fresh check against
  `pfn_variable_xdim_smoke.pt` after promotion, the production module still
  shows the same pattern (mean gap `-0.24` across a 10-instance, `k=3` run
  — always the joint score reading more optimistic than reality, never
  less). The production API makes the fix (independent standalone
  re-score) mandatory, not optional: `val_star`, returned by
  `kstep_explore_search`, IS the honest value by construction — there is no
  return path that hands back the inflated joint score under that name. In
  that same check, honest lookahead beat the 1-step search in 3/4
  signal-bearing instances (mean gain `+0.81`) and lost in 1/4 (`-2.01`) —
  small sample, but the losing case is kept, not filtered out, on purpose:
  k-step is not assumed to dominate.

  One design choice explicitly reconsidered during promotion: an early
  version (matching the notebook prototype) trust-region-constrained the
  plan's first point to stay near its seed ("nudge, don't replace"). Dropped
  before promotion — it risks trapping the search in whatever local basin
  the seed sits in, exactly when the objective wants a genuinely better,
  distant point. The production module has no such constraint, matching
  `explore_search`'s own (also-unconstrained) design.

## Conceptual status

The thing that actually matters, stated plainly: **does behavioral cloning
on privileged-search labels produce a policy that beats random search on
held-out incumbent-AUC?** Not yet, at either x_dim tested (1 or 6), even
after three separate, confirmed bug fixes along the way (a DAgger mixing-
schedule inversion, an earlier `x_realized`-seeding choice that made its own
target partly unlearnable, and a Beta-NLL loss floor that gave the network a
zero-cost "give up" point). This is **not a learnability failure** — the
network demonstrably learns to reproduce the oracle's own target faithfully
once those bugs were fixed — so the gap is somewhere else.

## Bottlenecks

In the order they're most likely blocking the conceptual result above —
later ones are only informative once earlier ones are addressed, not
because they matter less on their own:

1. **Proxy-filtered, but not regret-filtered, label quality — gate built,
   wired into the trainer and a push-button Hydra pipeline; end-to-end AUC
   effect on a real-scale run still open.**
   `build_explore_buffer` already discards corrections whose own weighted-
   NLL score didn't improve (`require_improvement=True`, existing default)
   — but the component-level table above shows that filter alone still lets
   through a large, variable fraction of corrections that don't reduce real
   regret. A genuine regret-based gate is now built:
   `build_explore_buffer(..., require_regret_improvement=True)`, using
   `search.explore.greedy_regret` (moved there from
   `pipelines/explore_search_playground.py` specifically so this trainer
   code could use it without a circular import) to gate on real regret
   improving, not the proxy. Costs two extra PFN forward passes per
   (instance, step) considered; off by default so existing behavior doesn't
   silently change. `ActionHeadImitationTrainer(require_regret_improvement=...)`
   and `configs/trainer/action_head_imitation_trainer.yaml`'s matching field
   expose it end to end; a push-button real-scale experiment
   (`configs/experiment/action_head_imitation_integrated_regret_kstep_real.yaml`,
   turns this on together with 2 and 3 below) is ready but **not yet run**.

   Measured directly, same 10-episode protocol as the table above
   (`pfn_variable_xdim_smoke.pt`, x_dim=6):

   | gate | corrections | mean regret before | mean regret after | mean reduction |
   |---|---|---|---|---|
   | proxy only (`require_improvement`) | 1429 | 0.137 | 0.070 | 0.067 |
   | + regret gate (`require_regret_improvement`) | 798 | 0.188 | 0.030 | **0.158** |

   The regret gate cuts the surviving-correction count by 44% and more than
   doubles the mean regret reduction per surviving correction (0.067 →
   0.158) — every surviving example strictly reduces regret and none
   worsen it *by construction* of the gate (100%/0%, not a new empirical
   finding, just confirmation the filter is implemented correctly). This is
   real, positive evidence the fix targets the right thing, but it is a
   buffer-level measurement, not the actual test: whether training on the
   smaller, cleaner buffer instead of the larger, noisier one moves
   `auc_improvement_vs_random` still requires an actual training run
   (`pipelines/train_exit.py`) with the gate on vs. off, not yet done —
   still the single most important open measurement in this file.
2. **k-step planning is now wired into the training loop, not just the
   label-generation toolkit.** `search/kstep_explore.py` exists and is
   tested; `build_explore_buffer(..., k=...)` now calls
   `kstep_explore_search` when `k>1` instead of `explore_search`, and
   `ActionHeadImitationTrainer(explore_k=...)` /
   `configs/trainer/action_head_imitation_trainer.yaml`'s `explore_k` field
   expose it end to end (default `1`, i.e. unchanged behavior, so this is
   opt-in). What's still open is empirical, not mechanical: whether `k>1`
   labels actually beat `k=1` on `auc_improvement_vs_random` — needs the
   real-scale run below.
3. **Seeding strategy now depends on which policy generated the rollout,
   opt-in.** Seeding the explore search at the incumbent is the right,
   safe choice under a random round-0 policy (seeding at the policy's own
   proposal there reproduces a real, previously-diagnosed collapse — the
   proposal is statistically independent of context under a random policy,
   so a correction anchored to it teaches "ignore your input"). Once a real
   policy exists to roll out with, its own proposal becomes genuinely
   context-dependent, and anchoring the search there instead becomes the
   more natural "correct what the policy actually did" signal for BC to
   improve on. `ActionHeadImitationTrainer(round_dependent_seeding=True,
   realized_seed_min_self_generated=0.7)` now switches `x_seed_mode` from
   `"incumbent"` to `"realized"` once `dagger/frac_self_generated` clears
   that threshold, logged as `explore/x_seed_mode_realized`; verified via a
   tiny push-button run (`trainer.n_rollouts=4`) actually flipping from 0.0
   to 1.0 across rollouts as `frac_self_generated` crossed 0.7. Still open:
   whether this measurably helps a real-scale trained policy, not just that
   the switch fires correctly.
4. **Single-shot, unchunked `ActionHead` vs. the now-wired-up chunked
   alternative.** The simple head is confirmed capable of finding a *given*
   argmax; not yet asked to represent or commit to a multi-step plan. A
   chunked, flow-matching alternative now has its own full push-button
   training pipeline (`models/action_head_flow.py`'s
   `FlowMatchingActionHead`, `trainer/action_head_flow_trainer.py`'s
   `ActionHeadFlowTrainer`, `pipelines/train_exit_flow.py`,
   `configs/train_exit_flow.yaml`) but is deliberately **not yet run at
   real scale or compared to the simple head**: doesn't block testing
   bottlenecks 1–3, and shouldn't be evaluated before them (see
   prioritization below) — a negative result on a bigger architecture can't
   be attributed to architecture vs. labels if the label problem underneath
   it is still unresolved.

## Prioritized experiments

All four of the following now have working, tested, push-button Hydra
pipelines (verified via CLI smoke runs and the full test suite, 171/171
passing) — none has been run at real scale yet. Per this project's own
established practice, none should be launched on `ulysses`/LUIS without
confirming with the user first (see `CLAUDE.md`).

1. **Label valuation/filtering — gate built, wired end to end, ready to
   run.** `build_explore_buffer(..., require_regret_improvement=True)`
   exists, tested, and shown to cut surviving corrections by 44% while more
   than doubling mean regret reduction per correction (see Bottleneck 1's
   table) — that was a buffer-level measurement; a real training run
   comparing `auc_improvement_vs_random` with it on vs. off is still the
   single most informative not-yet-run experiment in this file, since it
   directly tests the leading bottleneck hypothesis end to end. Run via
   `experiment=action_head_imitation_integrated_regret_kstep_real
   pfn_checkpoint=variable_xdim_smoke` (bundled with 2 and 3 below — see
   that config's own header comment for why they're combined rather than
   run as three separate ablations first).
2. **Promote k-step planning to production — done, wired into training.**
   `search/kstep_explore.py` exists, tested, with the standalone re-score
   mandatory (not a diagnostic someone could forget to call), and is now
   the label source `build_explore_buffer` uses whenever `k>1`. Still
   needed: comparing `k=1` vs. `k=3` labels on the AUC check from a real
   run (the same one as experiment 1, `explore_k: 3` in that config).
3. **Round-dependent seeding — done, wired into training.**
   Incumbent-anchored for round-0/random-policy rollouts,
   policy-proposal-anchored once `dagger/frac_self_generated` crosses
   `realized_seed_min_self_generated` (default `0.7`). Mechanically
   verified (the switch fires at the right threshold); not yet validated
   for whether it measurably helps — same real run as 1 and 2.
4. **Flow-matching / chunked `ActionHead` — full pipeline built, training
   run not started.** `FlowMatchingActionHead` (`models/action_head_flow.py`)
   trains (rectified-flow loss decreases on gradient steps, checked in its
   own demo and in a Hydra CLI smoke run —
   `experiment=action_head_flow_integrated_smoke`, `policy_loss/train`
   dropped 1.89→0.59 over 4 rollouts), samples valid `[0,1]^x_dim` chunks,
   and keeps the PFN gradient-isolated — confirmed by
   `tests/test_action_head_flow.py`, `tests/test_action_head_flow_trainer.py`,
   `tests/test_action_head_validation_callbacks.py`, and the CLI smoke run
   above (which also exercises `pipelines/train_exit_flow.py`,
   `configs/train_exit_flow.yaml`, and `build_flow_auc_eval_callback`
   end to end). Not yet trained against real `search.kstep_explore` chunk
   targets at scale or compared to the simple head on
   `auc_improvement_vs_random` — deliberately deferred until 1–3 produce a
   working, AUC-beating baseline with the current simple architecture on a
   real run, so a negative result here can be attributed to the
   architecture and not to labels inherited from an unresolved problem
   underneath it (the same confound the EI-argmax isolation diagnostic
   exists to avoid at a smaller scale). Real-scale config ready:
   `experiment=action_head_flow_integrated_real pfn_checkpoint=variable_xdim_smoke`.
