"""Explore-branch privileged search (M5) — gradient descent on a candidate
query `x_explore`, teacher-forced through the true BNN instance, scored by
the frozen PFN's weighted NLL at a fixed set of privileged "interesting"
points. Full design history and reasoning: `docs/log/2026-08-28-explore-search-input-optimization-and-teacher-forcing.md`
-- read that before changing this file, it records several dead ends
(self-referential y prediction, raw entropy) so they don't get silently
reintroduced.

Mechanism, precisely:

  1. `x_int`/`y_int_true` (from `interesting_points.build_interesting_points`)
     are FIXED for the whole episode -- Sobol + random + GD-restart-found
     basins on the true surface, computed once before rollout.
  2. At an explore-labeled rollout step (no incumbent improvement --
     `trainer.exit_rollout.label_branches`), weight each interesting point
     by how much true improvement it still represents over the CURRENT
     incumbent: `weight_i = max(0, log(incumbent) - log(y_int_true_i))`
     (`improvement_weights`, same log-improvement quantity
     `metrics/inc_auc.py`'s `log_incumbent_stepwise_reward` uses on a
     realized trajectory, applied here to a candidate set instead). Points
     already worse than the incumbent get weight 0 -- resolving uncertainty
     about them can't reduce regret.
  3. Optimize a candidate query `x_explore` (multistart, seeded at the
     incumbent -- see `explore_search`'s `x_seed` note, and
     `docs/log/2026-09-01-explore-branch-beta-nll-uniform-collapse.md` for
     why) to minimize `sum_i weight_i * NLL(y_int_true_i | PPD at x_int_i
     given x_explore added to context)`. `x_explore`'s y is teacher-forced from
     the true BNN instance (`prior.evaluate(x_explore, noise=False)`,
     recomputed fresh every GD step so it always tracks the *current*
     `x_explore` -- never a stale or self-predicted value) -- the same
     privileged, deterministic surface `search.exploit.exploit_search`
     already optimizes directly. `prior`'s own parameters are never
     learnable; `x_explore` is the only free tensor anywhere in this graph,
     and the BNN sits in the graph purely as a frozen-but-differentiable
     function grounding it in reality.

Why NLL, not entropy (2026-08-28, user-directed correction): entropy only
measures the PFN's own confidence, blind to whether that confidence is
correct -- a model can reduce entropy by becoming falsely sure of the wrong
answer. NLL against the *known* true `y_int_true` can't be gamed that way:
being confidently wrong makes NLL sharply worse, not better, while entropy
would reward it. This also uses the bar-distribution's native training
loss (`bar_dist.forward`, already implemented/tested) rather than a derived
quantity, so the search evaluates the model in the same currency it was
trained in.

Why weighting alone doesn't collapse this onto the exploit branch's target:
summing (weighted) NLL across the *whole* `x_int` set means the point that
best serves the objective need not be *at* any single high-weight point --
e.g. a point that disambiguates two nearby competing basins can dominate
even though neither individually has the highest weight. The real risk
instead is weight collapse: once the incumbent matches or beats every point
in `x_int` (which the exploit branch actively works toward), every weight
goes to 0 and the objective goes flat -- `explore_search` doesn't hide
this; see `has_signal` below and `trainer.exit_rollout.build_explore_buffer`,
which skips building a buffer entry when it fires. See the log entry above
for a more precise caveat: this collapse-avoidance argument weakens (though
doesn't vanish) as the weight distribution concentrates on one dominant
optimum.

What this does NOT fix: NLL is measured through the same, possibly
under-calibrated, frozen PFN as everything else here -- M2's own finding
that entropy doesn't shrink monotonically with context size on some
instances is a property of that PFN's calibration, not fixed by changing
the loss. It also doesn't address the deeper "input-optimization against a
frozen network" risk (gradient search finding out-of-training-distribution
`(x, y)` configurations that game the network rather than reflect it) --
see the log entry's "adversarial input optimization" section. Whether the
resulting corrections genuinely reduce *regret* (not just weighted NLL) is
checked empirically, not assumed -- see
`pipelines/explore_search_playground.py`, built specifically to verify this
against privileged ground truth before trusting this branch's outputs.
"""
import torch

from anytimeacquisition.models.bar_distribution import BarDistribution
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.priors.bnn import BNNPrior


def improvement_weights(incumbent: torch.Tensor, y_int_true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """weight_i = max(0, log(incumbent) - log(y_int_true_i)) -- same
    log-improvement quantity as `metrics.inc_auc.log_incumbent_stepwise_reward`,
    applied to a fixed candidate set instead of a realized trajectory.
    incumbent: [B]  y_int_true: [B, N_int] -> [B, N_int]."""
    return (torch.log(incumbent.clamp_min(eps)).unsqueeze(-1) - torch.log(y_int_true.clamp_min(eps))).clamp_min(0.0)


def _weighted_nll(
    pfn: PFN, bar_dist: BarDistribution, x_context: torch.Tensor, y_context: torch.Tensor,
    x_explore: torch.Tensor, y_explore_true: torch.Tensor, x_int: torch.Tensor, y_int_true: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """x_context/y_context: [B, Nt, x_dim]/[B, Nt] (already expanded to one
    row per restart by the caller). x_explore/y_explore_true: [B, 1, x_dim]/
    [B, 1] -- the candidate being optimized and its teacher-forced true y.
    x_int/y_int_true/weights: [B, N_int, x_dim]/[B, N_int]/[B, N_int].
    -> weighted NLL per row, [B] (lower is better, same convention entropy
    had). `bar_dist(logits, y)` is `BarDistribution.forward` -- the
    network's own NLL training loss, not a derived quantity."""
    x_train_aug = torch.cat([x_context, x_explore], dim=1)
    y_train_aug = torch.cat([y_context, y_explore_true], dim=1)
    logits_int = pfn(x_train_aug, y_train_aug, x_int)  # [B, N_int, n_bins]
    nll_int = bar_dist(logits_int, y_int_true)  # [B, N_int]
    return (weights * nll_int).sum(dim=-1)


def explore_search(
    prior: BNNPrior,
    pfn: PFN,
    bar_dist: BarDistribution,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_int: torch.Tensor,
    y_int_true: torch.Tensor,
    x_seed: torch.Tensor,
    n_restarts: int = 1,
    n_steps: int = 30,
    lr: float = 0.05,
    init_noise_std: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multistart GD on the frozen PFN's weighted NLL at `x_int` (see module
    docstring). `prior` must be the *same live instance* `x_context`/
    `y_context`/`x_int`/`y_int_true` came from — its `evaluate(..., noise=False)`
    is what teacher-forces `x_explore`'s y at every step (same
    never-reset-mid-episode requirement as `search.exploit.exploit_search`).
    `pfn` is frozen (params detached, eval mode) but NOT run under
    `torch.no_grad()` -- gradients must flow to `x_explore` through both
    `prior.evaluate` and the PFN's NLL computation.

    `x_seed` [B, x_dim]: where every restart is seeded — **the incumbent**
    (`x_context[argmin(y_context)]`, see `trainer/exit_rollout.py::build_explore_buffer`)
    as of 2026-09-01 (see `docs/log/2026-09-01-explore-branch-beta-nll-uniform-collapse.md`
    for the full investigation this replaces). Previously seeded at
    `x_realized`, the point the rollout's *own* policy actually played that
    step — reasonable in spirit ("correct the policy's own proposal," not
    an independent search), but under `random_policy` (round 0, and most of
    a `dagger_decay_rounds` run given how slowly beta decays) `x_realized`
    is literally `torch.rand(...)`, statistically independent of context.
    Measured directly: `corr(x_star, x_realized) = 0.77` across 1283
    explore-labeled examples, vs. `corr(x_star, incumbent_x) = -0.21` —
    x_star was dominated by a quantity the ActionHead structurally cannot
    observe (it only ever sees context), driving the trained policy to
    collapse to a context-independent uniform Beta (exactly zero NLL for
    any target when `alpha=beta=1`) rather than learning anything. The
    incumbent is context-visible (literally `y_context.argmin()`), so this
    removes that confound while keeping the same "stay local, don't
    roam blind on a tiny context" property `x_realized`-seeding had — it's
    a different context-anchored point, not a reversion to the earlier
    "fresh independent random draw" design already rejected once (see the
    log entry above and `docs/log/2026-08-28-explore-search-input-optimization-and-teacher-forcing.md`).
    Known trade-off, not fully resolved: this loses "correct what the
    policy actually proposed" for later, self-play-dominant rounds where
    `x_realized` would itself be context-derived and arguably a better
    anchor — deferred until the base learnability problem (this fix) is
    confirmed, not solved simultaneously.

    Note this is a *different* fixed point than seeding near the
    highest-weight `x_int` point, which an earlier design explicitly
    avoided ("collapse-toward-the-optimum risk the weighting avoids at the
    objective level" — see the module docstring's point 3): `x_int` are
    privileged points never in the training context, so seeding there
    would bias the search toward reproducing what the weighting already
    identifies as important, defeating its purpose. The incumbent is a
    training-context point used purely as a local-search starting position,
    same role `x_realized` played, not a proxy for "the answer."

    `n_restarts=1` by default (compute budget: each restart is a full
    `n_steps`-length GD trajectory through the PFN, and explore-labeled
    steps already vastly outnumber exploit-labeled ones per rollout) —
    restarts beyond the first, if used, get a small Gaussian position
    jitter (`init_noise_std`) around `x_seed`, not a global random
    location.

    -> (x_star [B, x_dim], weighted_nll_star [B], has_signal [B] bool --
    False where every interesting point's weight was already 0 for that
    instance (incumbent already matches/beats the whole x_int set) -- the
    returned x_star there is meaningless, caller must check this before use.
    `weighted_nll_star` is a raw value (not a before/after improvement) --
    callers wanting the improvement, or a genuine regret-based validation of
    what this search actually achieved, should use
    `pipelines/explore_search_playground.py`'s diagnostics rather than
    reading this number in isolation.
    """
    B, _, x_dim = x_context.shape

    pfn.eval()
    for p in pfn.parameters():
        p.requires_grad_(False)

    incumbent = y_context.min(dim=1).values  # [B]
    weights = improvement_weights(incumbent, y_int_true)  # [B, N_int]
    has_signal = weights.sum(dim=-1) > 0.0  # [B]

    x_context_rep = x_context.repeat_interleave(n_restarts, dim=0)  # [B*R, Nt, x_dim]
    y_context_rep = y_context.repeat_interleave(n_restarts, dim=0)  # [B*R, Nt]
    x_int_rep = x_int.repeat_interleave(n_restarts, dim=0)  # [B*R, N_int, x_dim]
    y_int_true_rep = y_int_true.repeat_interleave(n_restarts, dim=0)  # [B*R, N_int]
    weights_rep = weights.repeat_interleave(n_restarts, dim=0)  # [B*R, N_int]

    # Native [B, R, x_dim] shape for prior.evaluate (same multistart
    # convention as exploit_search's own `candidates` -- BNNPrior.evaluate
    # treats R as just more query points per instance, no repeat needed
    # here; only the PFN calls need the repeated-row [B*R, ...] form, since
    # each restart needs its own separate augmented-context forward pass).
    base = x_seed.unsqueeze(1).expand(B, n_restarts, x_dim)
    if n_restarts > 1 and init_noise_std > 0:
        jitter = torch.cat([
            torch.zeros(B, 1, x_dim),
            torch.randn(B, n_restarts - 1, x_dim) * init_noise_std,
        ], dim=1)
    else:
        jitter = torch.zeros(B, n_restarts, x_dim)
    candidates = (base + jitter).clamp(0.0, 1.0).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([candidates], lr=lr)

    # Best-so-far tracking, not just the final iterate: Adam's steps aren't
    # monotonically decreasing (momentum can overshoot, same issue
    # `search.exploit.exploit_search` guards against with its incumbent
    # fallback) -- but there's no privileged "known-good" x to fall back to
    # here (unlike exploit's context-derived incumbent), since weighted NLL
    # is evaluated through the PFN itself, not a true, known surface.
    # Tracking the best value actually visited at any step, for every
    # restart, is the direct fix: it can only ever match or improve on
    # "just take the final position".
    best_val = torch.full((B * n_restarts,), float("inf"))
    best_x = candidates.detach().reshape(B * n_restarts, 1, x_dim).clone()

    def _update_best(val: torch.Tensor, x_rep: torch.Tensor) -> None:
        nonlocal best_val, best_x
        with torch.no_grad():
            improved = val.detach() < best_val
            best_val = torch.where(improved, val.detach(), best_val)
            best_x = torch.where(improved.view(-1, 1, 1), x_rep.detach().clone(), best_x)

    for _ in range(n_steps):
        y_explore_true = prior.evaluate(candidates, noise=False)  # [B, R] -- teacher-forced, differentiable
        x_explore_rep = candidates.reshape(B * n_restarts, 1, x_dim)
        y_explore_true_rep = y_explore_true.reshape(B * n_restarts, 1)

        val = _weighted_nll(
            pfn, bar_dist, x_context_rep, y_context_rep, x_explore_rep, y_explore_true_rep,
            x_int_rep, y_int_true_rep, weights_rep,
        )
        _update_best(val, x_explore_rep)
        opt.zero_grad()
        val.sum().backward()
        opt.step()
        with torch.no_grad():
            candidates.clamp_(0.0, 1.0)

    with torch.no_grad():
        y_explore_true = prior.evaluate(candidates, noise=False)
        x_explore_rep = candidates.reshape(B * n_restarts, 1, x_dim)
        y_explore_true_rep = y_explore_true.reshape(B * n_restarts, 1)
        val_final = _weighted_nll(
            pfn, bar_dist, x_context_rep, y_context_rep, x_explore_rep, y_explore_true_rep,
            x_int_rep, y_int_true_rep, weights_rep,
        )
    _update_best(val_final, x_explore_rep)

    best_val = best_val.view(B, n_restarts)
    best_x = best_x.view(B, n_restarts, x_dim)
    best_idx = best_val.argmin(dim=1)
    x_star = best_x[torch.arange(B), best_idx]
    val_star = best_val[torch.arange(B), best_idx]
    return x_star, val_star, has_signal


if __name__ == "__main__":
    """M5.md's required concrete check: show weighted NLL at x_int actually
    decreasing once x_star is added to context, not just that the code
    runs. x_dim=1 -- the only trained checkpoint in this repo so far
    (`checkpoints/pfn_smoke_xdim1.pt`, a smoke-scale run, not a converged
    one). For a genuine regret-based validation (does this actually help
    find better points, not just reduce weighted NLL), see
    `pipelines/explore_search_playground.py`."""
    from anytimeacquisition.utils.paths import CHECKPOINT_DIR
    from anytimeacquisition.pipelines.train_pfn import load_pfn_checkpoint
    from anytimeacquisition.search.interesting_points import build_interesting_points

    torch.manual_seed(0)
    checkpoint_path = CHECKPOINT_DIR / "pfn_smoke_xdim1.pt"
    pfn, bar_dist, ckpt = load_pfn_checkpoint(checkpoint_path)
    x_dim = ckpt["config"]["max_x_dim"]
    print(f"loaded PFN checkpoint: {checkpoint_path.name}, config={ckpt['config']}")

    prior = BNNPrior(batch_size=3, x_dim=x_dim, seed=0)
    prior.reset()
    x_int, y_int_true = build_interesting_points(prior, n_sobol=16, n_random=16, n_basin_restarts=8)

    x_context, y_context, _, _ = prior.sample_episode(n_train=6, n_test=0)
    incumbent = y_context.min(dim=1).values
    weights = improvement_weights(incumbent, y_int_true)
    print("current incumbent per instance:            ", incumbent.tolist())
    print("interesting points with nonzero weight:    ", (weights > 0).sum(dim=1).tolist(), f"/ {x_int.shape[1]}")

    with torch.no_grad():
        nll_before = bar_dist(pfn(x_context, y_context, x_int), y_int_true)  # [B, N_int]
        weighted_nll_before = (weights * nll_before).sum(dim=-1)

    # Seed at the incumbent (context-visible) -- see trainer/exit_rollout.py::build_explore_buffer
    # and docs/log/2026-09-01-explore-branch-beta-nll-uniform-collapse.md for why.
    incumbent_idx = y_context.argmin(dim=1)
    x_seed = x_context[torch.arange(prior.B), incumbent_idx]
    x_star, val_star, has_signal = explore_search(
        prior, pfn, bar_dist, x_context, y_context, x_int, y_int_true, x_seed, n_restarts=1, n_steps=30, lr=0.05,
    )
    print("has_signal per instance (False = weights all zero, result meaningless):", has_signal.tolist())

    with torch.no_grad():
        y_star_true = prior.evaluate(x_star.unsqueeze(1), noise=False)  # [B, 1] -- teacher-forced, not the PFN's guess
        x_train_aug = torch.cat([x_context, x_star.unsqueeze(1)], dim=1)
        y_train_aug = torch.cat([y_context, y_star_true], dim=1)
        nll_after = bar_dist(pfn(x_train_aug, y_train_aug, x_int), y_int_true)
        weighted_nll_after = (weights * nll_after).sum(dim=-1)

    print("weighted NLL at x_int BEFORE adding x_star:", weighted_nll_before.tolist())
    print("weighted NLL at x_int AFTER  adding x_star:", weighted_nll_after.tolist())
    improved = 0
    for b in range(prior.B):
        if not has_signal[b]:
            print(f"  instance {b}: skipped (no signal)")
            continue
        improved += weighted_nll_after[b].item() <= weighted_nll_before[b].item() + 1e-4
    n_with_signal = int(has_signal.sum().item())
    print(f"weighted NLL at the interesting points decreased for {improved}/{n_with_signal} instances with signal "
          "(not asserted -- still reads through this checkpoint's own, possibly under-calibrated, PPD).")
