# Explore-search signal collapse + credit-assignment mechanism alternatives — open questions (2026-09-01)

Two problem statements from reviewing a Gemini conversation about the M5/
M5.5 design, deliberately recorded without picking a fix — both need more
thought before committing to a mechanism.

## 1. `has_signal=False` collapse in `search/explore.py`

`explore_search` gates on `has_signal = weights.sum(dim=-1) > 0`, where
`weights = improvement_weights(incumbent, y_int_true)` — this goes to zero
for every `x_int` point once the incumbent already matches/beats the
whole fixed test set for that instance/step.

This isn't just a labeling convenience. Once every weight is exactly 0,
`_weighted_nll`'s objective is *identically* 0 for every candidate
`x_explore` — the multistart GD loop in `explore_search` has zero gradient
everywhere, so whatever `x_star` it would return is arbitrary (whatever
the random init happened to be), not meaningful. `has_signal=False`
correctly detects and discards this degenerate case, but today only by
*skipping* the training example entirely
(`build_exploit_buffer`/`build_explore_buffer` in `trainer/exit_rollout.py`).

Skipping has a real cost worth naming: this collapse is most likely
exactly in the regime where the incumbent is already good (late-budget,
post-convergence) — meaning explore-branch training examples get
systematically sparser in precisely the phase a deployed policy would
spend a lot of its remaining budget in. `explore/signal_rate`
(`callbacks/action_head_validation.py`) exists to surface how often this
fires, but doesn't fix it.

Two candidate fix directions were considered and explicitly **not**
chosen — recorded as rejected/open, not a recommendation:

- Floor `improvement_weights` at a small `eps` instead of clamping to 0 —
  preserves the existing log-improvement scale's natural decay toward 0
  as the trajectory converges, but collapses to uniform (loses ranking
  among `x_int` points) once triggered.
- A rank-preserving softmax over `-α·y_true_i` — preserves ranking among
  candidates, but has a fixed total magnitude (softmax sums to 1)
  regardless of how converged the instance already is, unlike the natural
  quantity it would replace.

Whichever direction is eventually chosen will also need
`has_signal`/`explore/signal_rate` redefined (both candidate fixes make
`has_signal` basically always `True`), not just relaxed. What to actually
do about this is left open — this is a problem statement for future
reference, not a decision record.

## 2. Credit-assignment / instance-baselining mechanism alternatives (M5.5)

`docs/ROADMAP.md` Phase 5 already commits to per-trajectory,
per-instance-baseline AUC weighting for the imitation loss (not yet
implemented in `ActionHeadImitationTrainer` — a separate, already-known
gap, not part of this entry). Phase 5.5 plans a value-head-based advantage
estimate (`A_t = G_t - V_φ(s_t)`) as its first RL step.

This session's Gemini conversation proposed a heuristic, no-value-head
version of the same idea: a discounted per-step return
`G_t = Σ_k γ^(k-t) r_k` from the realized trajectory, baselined against
`μ_instance(G_t)` estimated from extra `random_policy` rollouts of the
same BNN instance. Discussed and deferred, per the user, in favor of
eventually building the real value head instead ("easier to sell" than a
heuristic proxy).

One further alternative mechanism is worth recording alongside it,
unresolved, for whenever M5.5 gets its real design review:

- **GRPO (Group Relative Policy Optimization)**: roll out `G` trajectories
  per BNN instance under the *current* policy, standardize each
  trajectory's total undiscounted return against the group's own
  mean/std (`A_i = (R_i - μ(R_1..G)) / σ(R_1..G)`) — no value network, no
  discount factor. Natively matches the "instance-baselining/luck
  control" already wanted. Note: needs `G` rollouts *under the current
  policy* (on-policy), not `random_policy` — a bigger shift from M5's
  round-0-only design than the original discounted-return proposal, and
  its usual LLM setting (`G` completions per prompt) maps to "`G`
  rollouts per BNN instance" here by analogy, not off-the-shelf.

Record this, the original discounted-return-with-baseline idea, and the
planned value head as open alternatives for the M5.5 design review — no
mechanism chosen here either.