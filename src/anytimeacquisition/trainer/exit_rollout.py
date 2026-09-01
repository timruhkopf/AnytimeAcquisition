"""EXIT self-play rollout + branch labeling — the data-generation half of
the imitation-learning loop from `docs/ROADMAP.md` Phase 5 / M5, per
`archive/src/exit/PFN_ActionHead_ExpertIteration_Design.md` §4's round
structure: self-play rollout -> branch labeling -> oracle correction ->
buffer of `(context, x*, y*)` tuples. This module builds the first three
steps; ActionHead retraining against the buffer is not built yet (see
"What's NOT here yet" below).

Keeping the live BNN instance across a rollout: `rollout_episode` calls
`prior.reset()` exactly once, at the start of the episode, and never again.
This is deliberate, not an oversight — the exploit branch's oracle search
(`search/exploit.py`) runs gradient descent directly on that *same instance's*
`evaluate(..., noise=False)` surface. A `reset()` anywhere mid-episode would
silently redraw a fresh random architecture/weights, so both the rollout's
own trajectory and any oracle search run against it would quietly start
optimizing a different ground-truth function than the one the context was
actually collected from.

Branch labeling reuses `metrics/inc_auc.py`'s `incumbent_trajectory` (the
same `cummin`-based running-best M3 already built for the log-incumbent AUC
metric) rather than reimplementing incumbent tracking — a rollout step is an
exploit-correction step iff the realized incumbent strictly improved at that
step (design doc §2 point 6: "partition by realized trajectory outcome, not
a learned rule").

Explore branch (2026-08-28, user-directed): explore-labeled (flat) steps get
an oracle target too now, via `build_explore_buffer` /
`search.explore.explore_search` — gradient descent on a candidate query,
teacher-forced through the true BNN, scored by the frozen PFN's weighted
NLL at a fixed, per-episode "interesting points" set
(`search/interesting_points.py`), weighted by privileged log-improvement
over the current incumbent. Full design history:
`docs/log/2026-08-28-explore-search-input-optimization-and-teacher-forcing.md`.
Needs a real PFN + BarDistribution (unlike the exploit branch, which only
needed the BNN) — pass `pfn`/`bar_dist` and `build_interesting_points_kwargs`
to use it; both default to `None`/off so existing exploit-only callers are
unaffected.

What's NOT here yet, deliberately (this is a skeleton, not the full M5
loop):
  - ActionHead retraining against the collected buffer, and the DAgger
    repeat-with-updated-policy loop around all of this
  - the round-0 self-play seeding question the design doc flags as open
    (§6) -- `random_policy` below (uniform random, ignores context) is the
    simplest reasonable placeholder until a trained ActionHead exists to
    roll out with instead
  - the multi-basin exploitation-pull fix from `docs/milestones/M5.md`
    (`exploit_search` keeps only its single best restart, not the full set
    of distinct basins found) -- the explore branch's weighted-sum-over-
    x_int objective is a different, complementary mitigation (see
    `search/explore.py`'s module docstring), not a substitute for this
"""
from dataclasses import dataclass

import torch

from anytimeacquisition.metrics.inc_auc import incumbent_trajectory
from anytimeacquisition.models.bar_distribution import BarDistribution
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.search.explore import explore_search, improvement_weights
from anytimeacquisition.search.exploit import exploit_search
from anytimeacquisition.search.interesting_points import build_interesting_points


@dataclass
class ImitationExample:
    """One raw `(context, x*, y*)` tuple for the DAgger-style aggregation
    buffer (design doc §2 point 7) — unbatched (one instance, one step),
    since which batch elements get labeled which branch varies per step.
    `instance_idx`/`step` identify where in the rollout this came from --
    exploit-labeled steps land at different rollout steps and in different
    counts per instance (whenever that instance's incumbent happened to
    improve), so without these two fields a flattened buffer can't be
    attributed back to "which instance, which point in its trajectory".

    `y_star`'s units are branch-dependent, matching the two oracles'
    genuinely different notions of "value" (design doc §2 point 7 leaves
    this ambiguous across branches, so being explicit here rather than
    silently conflating them): for `branch="exploit"` it's a true y (lower
    is better, same scale as `y_context`); for `branch="explore"` it's the
    achieved weighted-NLL objective value from `search.explore.explore_search`
    (also lower is better, but NOT a y value / not comparable across
    branches or to `y_context`)."""
    x_context: torch.Tensor  # [Nt, x_dim]
    y_context: torch.Tensor  # [Nt]
    x_star: torch.Tensor  # [x_dim]
    y_star: torch.Tensor  # scalar -- see branch-dependent units note above
    branch: str  # "exploit" | "explore"
    instance_idx: int
    step: int  # rollout step (0-indexed within the n_steps policy-chosen queries, not counting n_init)


def random_policy(x_context: torch.Tensor, y_context: torch.Tensor, x_dim: int) -> torch.Tensor:
    """Round-0 self-play seeding placeholder (design doc §6, `docs/milestones/M5.md`):
    uniform-random next query, ignoring context entirely. Stand-in until a
    trained ActionHead exists to roll out under instead."""
    B = x_context.shape[0]
    return torch.rand(B, x_dim)


def mixed_policy_fn(policy_a, policy_b, beta: float, usage_counter: dict | None = None):
    """DAgger-style mixture policy (Ross et al.) -- at every call (rollout
    step), each batch *instance* independently uses `policy_a` with
    probability `beta`, else `policy_b`. Per-instance rather than per-whole-
    rollout: a single rollout then already contains a mix of both kinds of
    transitions at a given step, not just across time -- matching DAgger's
    actual per-timestep mixture more closely than an all-or-nothing per-
    rollout coin flip would.

    Both `policy_a`/`policy_b` are called every time regardless of `beta`
    (simpler than trying to only compute the one that's "used", and cheap
    relative to everything else in a rollout step -- one extra PFN forward
    pass, not another oracle search) -- `torch.where` then selects per
    instance. Safe to use with a policy_fn that tracks internal state across
    calls (e.g. `models.action_head.action_head_policy_fn`'s step counter):
    both callables are still invoked exactly once per rollout step, exactly
    as `rollout_episode` already guarantees.

    `usage_counter`: optional mutable dict, updated in place with running
    `{"a": n, "b": n}` instance-counts across every call -- lets a caller
    read back the EMPIRICALLY realized `policy_a`/`policy_b` split after a
    whole rollout (`trainer.action_head_imitation_trainer`'s
    `dagger/frac_self_generated`), as a sanity check that the per-instance
    mixing is actually behaving like the intended `beta`, not just trusting
    the schedule blindly.

    See `trainer.action_head_imitation_trainer.ActionHeadImitationTrainer`
    for the per-round `beta` decay schedule this is built for.
    """
    def policy_fn(x_context: torch.Tensor, y_context: torch.Tensor, x_dim: int) -> torch.Tensor:
        action_a = policy_a(x_context, y_context, x_dim)
        action_b = policy_b(x_context, y_context, x_dim)
        use_a = torch.rand(x_context.shape[0]) < beta
        if usage_counter is not None:
            usage_counter["a"] = usage_counter.get("a", 0) + int(use_a.sum().item())
            usage_counter["b"] = usage_counter.get("b", 0) + int((~use_a).sum().item())
        return torch.where(use_a.unsqueeze(-1), action_a, action_b)
    return policy_fn


def rollout_episode(
    prior: BNNPrior, n_init: int, n_steps: int, policy_fn=random_policy, noise: bool = True,
    build_interesting_points_kwargs: dict | None = None, reset: bool = True,
) -> dict:
    """One self-play episode against `prior` — resets it once (fresh
    architecture/weights), then never again for the rest of the episode.
    `noise`: whether the *trajectory's own* observations are noisy (realistic
    self-play, the default) — independent of `exploit_search`'s own
    noise=False oracle access to the clean surface.

    `reset`: set False to roll out against `prior`'s *current* instance
    without redrawing it — the caller must have already called
    `prior.reset()` itself. For generating several independent trajectories
    against the exact same underlying BNN instance (e.g. averaging a cheap
    policy like `random_policy` over repeated restarts on one instance to
    get a lower-variance baseline estimate, `callbacks/action_head_validation.py`'s
    `build_auc_eval_callback`) — each call still draws its own fresh
    `x_context`/policy randomness, only the instance itself (its drawn
    architecture/weights) stays fixed across calls. Default `True` (reset
    every call) matches every existing caller's behavior unchanged.

    `build_interesting_points_kwargs`: if given (not None), builds the
    explore branch's fixed test-point set (`interesting_points.build_interesting_points`)
    right after this call's own `reset()` and before any policy step runs —
    the only place that can guarantee `x_int`/`y_int_true` come from the
    exact same instance draw used for the rest of the episode, per the
    "test tokens kept constant, computed prior to rollout" requirement
    (`search/explore.py`). Omit (default) to skip — old callers doing
    exploit-only rollouts are unaffected.

    -> {"x_context": [B, n_init+n_steps, x_dim], "y_context": [B, n_init+n_steps],
        "pre_step_contexts": [(x[B,Nt,x_dim], y[B,Nt]), ...] of length
        n_steps -- the context as it stood *immediately before* each rollout
        step, i.e. what a policy or an oracle search actually conditions on
        at that state. `branch labeling compares the realized outcome of
        acting from state i to state i's own pre-step context, per
        `label_branches` below.
        "x_int"/"y_int_true": present only if `build_interesting_points_kwargs`
        was given.}
    """
    if reset:
        prior.reset()
    result = {}
    if build_interesting_points_kwargs is not None:
        x_int, y_int_true = build_interesting_points(prior, **build_interesting_points_kwargs)
        result["x_int"], result["y_int_true"] = x_int, y_int_true

    x_context, y_context, _, _ = prior.sample_episode(n_train=n_init, n_test=0)

    pre_step_contexts = []
    for _ in range(n_steps):
        pre_step_contexts.append((x_context, y_context))
        x_next = policy_fn(x_context, y_context, prior.d).unsqueeze(1)  # [B,1,x_dim]
        y_next = prior.evaluate(x_next, noise=noise)  # [B,1]
        x_context = torch.cat([x_context, x_next], dim=1)
        y_context = torch.cat([y_context, y_next], dim=1)

    result["x_context"], result["y_context"], result["pre_step_contexts"] = x_context, y_context, pre_step_contexts
    return result


def label_branches(y_context: torch.Tensor, n_init: int) -> torch.Tensor:
    """-> [B, n_steps] bool, True at rollout steps where the realized
    incumbent strictly improved (exploit-correction steps; False = flat =
    explore-correction steps, not yet acted on here). Reuses
    `incumbent_trajectory` rather than recomputing running-best by hand."""
    inc = incumbent_trajectory(y_context, minimize=True)  # [B, n_init+n_steps]
    return inc[:, n_init:] < inc[:, n_init - 1:-1]


def build_exploit_buffer(
    prior: BNNPrior, rollout: dict, n_init: int, exploit_search_kwargs: dict | None = None,
    steps: set[int] | None = None, require_exploit_label: bool = True,
) -> list[ImitationExample]:
    """Runs `exploit_search` at every exploit-labeled (instance, step) pair
    and collects the resulting oracle corrections into a flat buffer. Calls
    `exploit_search` once per step across the *full* batch even when only
    some instances are exploit-labeled at that step (`BNNPrior.evaluate`
    requires its leading dim to exactly match the instance it was `reset()`
    with, so a labeled subset can't be sliced out and searched on its own
    without also slicing the prior's internal weight tensors — out of scope
    for this skeleton) — simpler and correct, at the cost of some wasted
    search on non-exploit instances.

    `steps`: optional allowlist of step indices to consider at all (other
    steps are skipped regardless of labeling) -- unset (default) considers
    every step, unchanged from before.

    `require_exploit_label`: `True` (default) matches every previous
    caller -- only instances `label_branches` actually calls exploit-labeled
    at a given step get a target. `False` flips the per-step mask to the
    *complement* (`~is_exploit`, i.e. the flat instances) instead -- used to
    generate "filler" exploit targets at steps the explore branch chose not
    to spend its own budget on (`trainer.action_head_imitation_trainer`'s
    `fill_unselected_explore_steps_with_exploit`, 2026-09-01), rather than
    leaving those instances with no training example at all. Exploit's own
    oracle (a local refinement of the current incumbent, given the current
    context) is well-defined at ANY step, not just ones where the realized
    trajectory happened to already show an improvement -- `label_branches`
    is a labeling/routing choice for which oracle to consult by default,
    not a claim that `exploit_search`'s result is only meaningful at
    naturally-improving steps. A caller combining a `require_exploit_label=True`
    call with a `require_exploit_label=False` call at *different, disjoint*
    step sets never double-labels the same (instance, step) pair (each
    instance is exploit-labeled XOR flat at a given step, never both).
    """
    exploit_search_kwargs = exploit_search_kwargs or {}
    is_exploit = label_branches(rollout["y_context"], n_init)  # [B, n_steps]

    buffer: list[ImitationExample] = []
    for step, (x_ctx, y_ctx) in enumerate(rollout["pre_step_contexts"]):
        if steps is not None and step not in steps:
            continue
        step_mask = is_exploit[:, step] if require_exploit_label else ~is_exploit[:, step]
        if not step_mask.any():
            continue
        x_star, y_star = exploit_search(prior, x_ctx, y_ctx, **exploit_search_kwargs)
        for b in torch.nonzero(step_mask, as_tuple=False).squeeze(-1).tolist():
            buffer.append(ImitationExample(
                x_context=x_ctx[b].clone(), y_context=y_ctx[b].clone(),
                x_star=x_star[b].detach().clone(), y_star=y_star[b].detach().clone(),
                branch="exploit", instance_idx=b, step=step,
            ))
    return buffer


def build_explore_buffer(
    prior: BNNPrior, pfn: PFN, bar_dist: BarDistribution, rollout: dict, n_init: int,
    explore_search_kwargs: dict | None = None, steps: set[int] | None = None,
) -> list[ImitationExample]:
    """Runs `explore_search` at every explore-labeled (instance, step) pair
    (the complement of `label_branches` -- flat steps) and collects the
    resulting oracle corrections into a flat buffer, same shape/reasoning as
    `build_exploit_buffer` (full-batch calls, subset-by-mask afterward).
    `prior` must be the same live instance `rollout` came from -- it's what
    teacher-forces each candidate `x_explore`'s y during the search (see
    `search.explore.explore_search`'s module docstring).

    Requires `rollout` to have been built with `build_interesting_points_kwargs`
    set on `rollout_episode` — `x_int`/`y_int_true` must be present and,
    critically, must be the SAME fixed set used at every step of this
    episode (see `rollout_episode`'s docstring and `search/explore.py`'s
    module docstring for why).

    Skips any (instance, step) pair where `explore_search` reports
    `has_signal=False` (every interesting point's weight was already 0 for
    that instance at that step, see `search.explore.explore_search`) --
    there is nothing informative to imitate there, so no buffer entry is
    added for it, same as `build_exploit_buffer` skipping steps with no
    exploit-labeled instances at all.

    `explore_search` seeds its search at the point the rollout's own
    policy actually played at each step (`rollout["x_context"][:,
    n_init+step, :]`, `x_realized` below) rather than a fresh random draw
    -- a genuine correction of what was played, not an independent oracle
    search that ignores it (see `search.explore.explore_search`'s
    docstring, 2026-09-01).

    `steps`: optional allowlist of step indices to consider at all (other
    steps are skipped regardless of labeling) -- unset (default) considers
    every step, unchanged from before. Used by
    `trainer.action_head_imitation_trainer`'s `max_explore_steps_per_rollout`
    to bound how many (expensive, PFN-forward/backward-bearing)
    `explore_search` calls a single rollout pays for.
    """
    assert "x_int" in rollout and "y_int_true" in rollout, (
        "rollout must be built with rollout_episode(..., build_interesting_points_kwargs=...) "
        "to use the explore branch"
    )
    explore_search_kwargs = explore_search_kwargs or {}
    is_exploit = label_branches(rollout["y_context"], n_init)  # [B, n_steps]
    is_explore = ~is_exploit

    buffer: list[ImitationExample] = []
    for step, (x_ctx, y_ctx) in enumerate(rollout["pre_step_contexts"]):
        if steps is not None and step not in steps:
            continue
        step_mask = is_explore[:, step]
        if not step_mask.any():
            continue
        x_realized = rollout["x_context"][:, n_init + step, :]  # [B, x_dim] -- the point actually played at this step
        x_star, val_star, has_signal = explore_search(
            prior, pfn, bar_dist, x_ctx, y_ctx, rollout["x_int"], rollout["y_int_true"], x_realized,
            **explore_search_kwargs,
        )
        for b in torch.nonzero(step_mask & has_signal, as_tuple=False).squeeze(-1).tolist():
            buffer.append(ImitationExample(
                x_context=x_ctx[b].clone(), y_context=y_ctx[b].clone(),
                x_star=x_star[b].detach().clone(), y_star=val_star[b].detach().clone(),
                branch="explore", instance_idx=b, step=step,
            ))
    return buffer


def plot_exploit_corrections_over_trajectory(
    prior: BNNPrior, buffer: list["ImitationExample"], instance_idx: int, grid_res: int = 100, ax=None,
):
    """2D-only diagnostic: instance `instance_idx`'s true surface with every
    exploit-branch oracle correction found across a *full rollout* overlaid,
    colored/labeled by rollout step -- shows `exploit_search` staying
    correct across a whole trajectory's incumbent-improvement steps, which
    land at different rollout steps and in different counts per instance
    (not a single fixed context, unlike `search.exploit.plot_restart_trajectories`).
    Returns the Axes.
    """
    import matplotlib.pyplot as plt

    assert prior.d == 2, "plot_exploit_corrections_over_trajectory is 2D-only"
    examples = [ex for ex in buffer if ex.instance_idx == instance_idx]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ink = "#1a1a1a"
    if not examples:
        ax.set_title(f"instance {instance_idx}: no exploit-labeled steps in this rollout", fontsize=10, color=ink)
        return ax

    lin = torch.linspace(0.0, 1.0, grid_res)
    grid = torch.stack(torch.meshgrid(lin, lin, indexing="ij"), dim=-1).reshape(1, -1, 2).expand(prior.B, -1, -1)
    with torch.no_grad():
        grid_y = prior.evaluate(grid, noise=False)[instance_idx].reshape(grid_res, grid_res)
    im = ax.contourf(lin.numpy(), lin.numpy(), grid_y.numpy().T, levels=30, cmap="viridis")
    ax.figure.colorbar(im, ax=ax, pad=0.02).set_label("true y (lower is better)", color=ink)

    steps = [ex.step for ex in examples]
    cmap = plt.get_cmap("plasma")
    lo, hi = min(steps), max(steps)
    norm = plt.Normalize(lo, hi if hi > lo else lo + 1)
    for ex in sorted(examples, key=lambda e: e.step):
        color = cmap(norm(ex.step))
        x_star_np = ex.x_star.numpy()
        ax.scatter(*x_star_np, marker="X", s=150, color=color, edgecolor="white", linewidth=1.0, zorder=5)
        ax.annotate(str(ex.step), x_star_np, textcoords="offset points", xytext=(6, 6), fontsize=8, color=ink)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x1", color=ink)
    ax.set_ylabel("x2", color=ink)
    ax.set_title(
        f"instance {instance_idx}: {len(examples)} exploit corrections across the rollout "
        f"(label = rollout step)", fontsize=10, color=ink,
    )
    return ax


if __name__ == "__main__":
    """Demo: one self-play episode on a live BNNPrior, showing (a) the
    trainer detects incumbent changes via the reused inc-AUC code, and (b)
    the exploit branch's oracle correction is a real, live-instance search
    that finds points at least as good as -- typically better than -- what
    the trajectory's own current incumbent had reached. No PFN or ActionHead
    involved: this validates the BNN-side machinery (rollout state-keeping,
    branch labeling, exploit search) entirely on its own."""
    torch.manual_seed(0)
    x_dim = 2
    n_init, n_steps = 5, 15
    prior = BNNPrior(batch_size=4, x_dim=x_dim, seed=0)

    rollout = rollout_episode(prior, n_init=n_init, n_steps=n_steps, policy_fn=random_policy)
    inc = incumbent_trajectory(rollout["y_context"], minimize=True)
    print("incumbent trajectory, instance 0:", [f"{v:.4f}" for v in inc[0].tolist()])

    is_exploit = label_branches(rollout["y_context"], n_init)
    print(f"exploit-labeled steps per instance: {is_exploit.sum(dim=1).tolist()} / {n_steps}")
    for b in range(prior.B):
        step_idx = torch.nonzero(is_exploit[b], as_tuple=False).squeeze(-1).tolist()
        print(f"  instance {b}: incumbent improved at rollout steps {step_idx}")

    buffer = build_exploit_buffer(prior, rollout, n_init, exploit_search_kwargs={"n_restarts": 12, "n_steps": 60})
    print(f"buffer size: {len(buffer)} exploit-correction examples")

    beats_incumbent = 0
    for ex in buffer:
        current_incumbent = ex.y_context.min().item()
        beats_incumbent += ex.y_star.item() <= current_incumbent + 1e-4
        assert ex.y_star.item() <= current_incumbent + 1e-4, (
            "oracle correction must never be worse than the context's own current incumbent"
        )
    print(f"oracle correction <= current incumbent: {beats_incumbent}/{len(buffer)} (expect all)")

    strictly_better = sum(ex.y_star.item() < ex.y_context.min().item() - 1e-4 for ex in buffer)
    print(f"oracle correction strictly better than current incumbent: {strictly_better}/{len(buffer)}")

    if x_dim == 2:
        from pathlib import Path

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, prior.B, figsize=(6 * prior.B, 5))
        for b, ax in enumerate(axes if prior.B > 1 else [axes]):
            plot_exploit_corrections_over_trajectory(prior, buffer, instance_idx=b, ax=ax)
        fig.tight_layout()
        out_path = Path(__file__).parent / "_demo_plots" / "exploit_corrections_over_trajectory.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print("saved", out_path)

    # --- Explore branch: needs a real PFN checkpoint, unlike exploit above,
    # which only needed the BNN's own surface. Separate x_dim=1 instance --
    # the only trained checkpoint in this repo so far uses x_dim=1.
    print("\n--- explore branch (from the realized trajectory, on the same rollout machinery) ---")
    from anytimeacquisition.utils.paths import CHECKPOINT_DIR
    from anytimeacquisition.pipelines.train_pfn import load_pfn_checkpoint

    checkpoint_path = CHECKPOINT_DIR / "pfn_smoke_xdim1.pt"
    pfn, bar_dist, ckpt = load_pfn_checkpoint(checkpoint_path)
    print(f"loaded PFN checkpoint: {checkpoint_path.name}, config={ckpt['config']}")

    explore_prior = BNNPrior(batch_size=4, x_dim=1, seed=1)
    explore_rollout = rollout_episode(
        explore_prior, n_init=n_init, n_steps=n_steps, policy_fn=random_policy,
        build_interesting_points_kwargs={"n_sobol": 16, "n_random": 16, "n_basin_restarts": 8},
    )
    is_exploit_e = label_branches(explore_rollout["y_context"], n_init)
    is_explore_e = ~is_exploit_e
    print(f"explore-labeled steps per instance: {is_explore_e.sum(dim=1).tolist()} / {n_steps}")

    explore_buffer = build_explore_buffer(
        explore_prior, pfn, bar_dist, explore_rollout, n_init, explore_search_kwargs={"n_restarts": 8, "n_steps": 30},
    )
    print(f"explore buffer size: {len(explore_buffer)} correction examples "
          f"(<= explore-labeled count -- zero-signal steps are skipped)")

    # NOT a hard per-example assert, deliberately -- unlike exploit_search's
    # guarantee (a real known y to fall back to), "weighted NLL" is only
    # ever as good as this *specific checkpoint's* calibration.
    # pfn_smoke_xdim1.pt is a smoke-scale run, and M2 already found its
    # entropy doesn't shrink monotonically with context size on some
    # instances (docs/log/2026-08-28-m2-pfn-and-bar-distribution.md) -- a
    # few explore corrections not helping is exactly that finding showing
    # up here, not a bug in this search. For a genuine regret-based check
    # (not just weighted NLL), see pipelines/explore_search_playground.py.
    improved_count = 0
    x_int, y_int_true = explore_rollout["x_int"], explore_rollout["y_int_true"]
    for ex in explore_buffer:
        incumbent = ex.y_context.min()
        weights = improvement_weights(incumbent.unsqueeze(0), y_int_true[ex.instance_idx].unsqueeze(0))[0]
        y_int_true_b = y_int_true[ex.instance_idx].unsqueeze(0)
        with torch.no_grad():
            nll_before = bar_dist(
                pfn(ex.x_context.unsqueeze(0), ex.y_context.unsqueeze(0), x_int[ex.instance_idx].unsqueeze(0)),
                y_int_true_b,
            )[0]
            weighted_before = (weights * nll_before).sum()
        improved_count += ex.y_star.item() <= weighted_before.item() + 1e-4
    print(f"explore corrections that reduced weighted NLL vs. doing nothing: "
          f"{improved_count}/{len(explore_buffer)}")
