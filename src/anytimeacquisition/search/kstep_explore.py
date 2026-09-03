"""k-step privileged planning — jointly optimize a short plan of `k` future
points against the frozen PFN's weighted NLL at `x_int`, keep only the
first point, discard the rest (receding-horizon / MPC: replan fresh from
the next real context, never blindly continue a stale plan). See
`docs/ROADMAP.md`'s ground-truth-privileged-search lens for why this is a
first-class generalization of `search.explore.explore_search`, not a
separate idea, and `docs/MILESTONES.md` for what's actually been measured
about it so far.

Promoted here from a notebook prototype
(`notebooks/kstep_explore_search_labeling.ipynb`, kept for the worked
demonstration/plots) once its central finding held up under measurement:
**a joint k-point plan's score is the team's value, not any one point's
own.** The frozen PFN reads context as a permutation-invariant set — it has
no notion of "point 1 arrived before point 2" — so the plan's reported
objective value cannot be decomposed back into per-point credit after the
fact (the same problem Shapley values exist to solve for jointly-produced
value in cooperative game theory). Measured directly, trusting the joint
score as the selected point's own value overstated its real, independently
-rescored contribution by as much as 17 (weighted-NLL units) in one case,
including cases where the point turned out to be no better than a naive
baseline.

**API consequence, and the reason this module exists as more than a
notebook:** `kstep_explore_search` below computes the standalone re-score
of `x_star` internally and *unconditionally* — there is no way to call this
function and get back an unverified, inflated value under the name
`val_star`. That name means exactly what it means for
`search.explore.explore_search`: `x_star`'s own, honest, independently
-rescored value, safe to use as-is (e.g. as a label-quality gate, matching
`trainer.exit_rollout.build_explore_buffer`'s `require_regret_improvement`
pattern). The joint plan's own (systematically optimistic) score is still
returned, under the clearly different name `joint_val`, for diagnostics
only — never meant to be read as "how good `x_star` is."
"""
import torch
from typing import Callable

from anytimeacquisition.models.bar_distribution import BarDistribution
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.search.explore import _weighted_nll, improvement_weights


def kstep_explore_search(
    prior: BNNPrior,
    pfn: PFN,
    bar_dist: BarDistribution,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_int: torch.Tensor,
    y_int_true: torch.Tensor,
    x_seed: torch.Tensor,
    k: int = 3,
    weight_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = improvement_weights,
    n_restarts: int = 4,
    n_steps: int = 40,
    lr: float = 0.05,
    record_trajectory: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Jointly gradient-descends a plan of `k` future points `(x_1,...,x_k)`
    against ONE shared objective — weighted NLL at `x_int`, computed with
    all `k` points hypothetically added to context at once, through a
    SINGLE PFN call per gradient step (this is what keeps k-step planning
    cheap: it costs about what `explore_search` already costs, just with a
    slightly longer context per call, NOT `k` separate PFN calls per
    gradient step).

    -- WHY SLOT 0 IS SPECIAL, AND WHY IT'S NOT TRUST-REGION-CONSTRAINED --
    `x_1` (slot 0 of the plan) is the only point ever actually deployed —
    `x_2,...,x_k` exist purely as imagined future leverage to make `x_1`'s
    choice smarter, then get discarded. Slot 0 starts exactly at `x_seed`
    (same multistart-diversity role `explore_search`'s own `x_seed` plays)
    but is NOT bounded to stay near it -- a trust region around the seed
    was tried and deliberately dropped: it risks trapping the search in
    whatever local basin the seed happens to sit in, exactly when the
    objective wants to reach a genuinely better, distant point. Slots
    1..k-1 start at independent random points, not at `x_seed` too — if
    every slot started identical, the (permutation-symmetric-in-x_2..x_k)
    objective would give every slot an identical gradient at every step,
    collapsing them into one point instead of exploring genuinely different
    roles.

    `x_seed` [B, x_dim]: same role as `explore_search`'s own `x_seed` — see
    that function's docstring for the current guidance on what to seed at
    and why (context-visible incumbent vs. the rollout's own realized
    action, and the round-dependent tradeoff between them).

    -> (x_star [B, x_dim] — the one point to actually use; val_star [B] —
        x_star's OWN honest, standalone-rescored weighted NLL, safe to use
        exactly like `explore_search`'s own `val_star`; has_signal [B];
        plan_star [B, k, x_dim] — the full best plan (`plan_star[:, 0] ==
        x_star`), inspection/plotting only, never meant to be executed;
        joint_val [B] — the joint plan's own score, diagnostics only, NEVER
        safe to read as `x_star`'s value — see module docstring), plus
        `trajectory` [n_steps+1, B, n_restarts, k, x_dim] if
        `record_trajectory=True`.
    """
    B, _, x_dim = x_context.shape
    pfn.eval()
    for p in pfn.parameters():
        p.requires_grad_(False)

    incumbent = y_context.min(dim=1).values
    weights = weight_fn(incumbent, y_int_true)
    has_signal = weights.sum(dim=-1) > 0.0

    x_context_rep = x_context.repeat_interleave(n_restarts, dim=0)  # [B*R, Nt, x_dim]
    y_context_rep = y_context.repeat_interleave(n_restarts, dim=0)  # [B*R, Nt]
    x_int_rep = x_int.repeat_interleave(n_restarts, dim=0)  # [B*R, N_int, x_dim]
    y_int_true_rep = y_int_true.repeat_interleave(n_restarts, dim=0)  # [B*R, N_int]
    weights_rep = weights.repeat_interleave(n_restarts, dim=0)  # [B*R, N_int]

    # candidates: [B, n_restarts, k, x_dim] -- k free points per restart.
    # Slot 0 anchored at x_seed (+ tiny restart jitter); slots 1..k-1 seeded
    # independently at random (see docstring -- symmetry breaking).
    slot0 = x_seed.unsqueeze(1).expand(B, n_restarts, x_dim).clone().unsqueeze(2)  # [B, R, 1, x_dim]
    if n_restarts > 1:
        slot0 = slot0.clone()
        slot0[:, 1:] = slot0[:, 1:] + torch.randn(B, n_restarts - 1, 1, x_dim) * 0.02
    other_slots = torch.rand(B, n_restarts, k - 1, x_dim) if k > 1 else torch.empty(B, n_restarts, 0, x_dim)
    candidates = torch.cat([slot0, other_slots], dim=2).clamp(0.0, 1.0)  # [B, R, k, x_dim]
    candidates = candidates.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([candidates], lr=lr)
    trajectory = [candidates.detach().clone()] if record_trajectory else None

    best_val = torch.full((B * n_restarts,), float("inf"))
    best_plan = candidates.detach().reshape(B * n_restarts, k, x_dim).clone()

    def _update_best(val: torch.Tensor, plan_rep: torch.Tensor) -> None:
        nonlocal best_val, best_plan
        with torch.no_grad():
            improved = val.detach() < best_val
            best_val = torch.where(improved, val.detach(), best_val)
            best_plan = torch.where(improved.view(-1, 1, 1), plan_rep.detach().clone(), best_plan)

    for _ in range(n_steps):
        # All k points' true y's, teacher-forced fresh from the current
        # candidates -- prior.evaluate treats "R*k points per instance" as
        # just more query points, the same multistart trick used elsewhere,
        # no special-casing needed for the extra k axis.
        y_true = prior.evaluate(candidates.reshape(B, n_restarts * k, x_dim), noise=False)
        y_true = y_true.reshape(B, n_restarts, k)

        plan_rep = candidates.reshape(B * n_restarts, k, x_dim)
        y_true_rep = y_true.reshape(B * n_restarts, k)

        val = _weighted_nll(
            pfn, bar_dist, x_context_rep, y_context_rep, plan_rep, y_true_rep,
            x_int_rep, y_int_true_rep, weights_rep,
        )
        _update_best(val, plan_rep)
        opt.zero_grad()
        val.sum().backward()
        opt.step()
        with torch.no_grad():
            candidates.clamp_(0.0, 1.0)
        if record_trajectory:
            trajectory.append(candidates.detach().clone())

    with torch.no_grad():
        y_true = prior.evaluate(candidates.reshape(B, n_restarts * k, x_dim), noise=False)
        y_true = y_true.reshape(B, n_restarts, k)
        plan_rep = candidates.reshape(B * n_restarts, k, x_dim)
        y_true_rep = y_true.reshape(B * n_restarts, k)
        val_final = _weighted_nll(
            pfn, bar_dist, x_context_rep, y_context_rep, plan_rep, y_true_rep,
            x_int_rep, y_int_true_rep, weights_rep,
        )
    _update_best(val_final, plan_rep)

    best_val = best_val.view(B, n_restarts)
    best_plan = best_plan.view(B, n_restarts, k, x_dim)
    best_idx = best_val.argmin(dim=1)
    plan_star = best_plan[torch.arange(B), best_idx]  # [B, k, x_dim]
    joint_val = best_val[torch.arange(B), best_idx]  # [B] -- the JOINT plan's score, diagnostics only
    x_star = plan_star[:, 0]  # [B, x_dim] -- the only point ever deployed

    # Mandatory standalone re-score -- NOT optional, this is the whole point
    # of this module existing separately from the notebook prototype it was
    # promoted from. x_star's own true y, teacher-forced fresh (not read
    # off the plan's own optimization, which used a possibly-stale y from
    # an earlier gradient step).
    with torch.no_grad():
        y_star_true = prior.evaluate(x_star.unsqueeze(1), noise=False)  # [B, 1]
        x_context_aug = torch.cat([x_context, x_star.unsqueeze(1)], dim=1)
        y_context_aug = torch.cat([y_context, y_star_true], dim=1)
        logits_star = pfn(x_context_aug, y_context_aug, x_int)
        nll_star = bar_dist(logits_star, y_int_true)
        val_star = (weights * nll_star).sum(dim=-1)  # [B] -- x_star's OWN honest value

    if record_trajectory:
        return x_star, val_star, has_signal, plan_star, joint_val, torch.stack(trajectory, dim=0)
    return x_star, val_star, has_signal, plan_star, joint_val


if __name__ == "__main__":
    """Demo + the concrete check this module's own docstring makes a claim
    about: does trusting the joint plan's score overstate x_star's real
    value, and does k-step lookahead honestly (via val_star, not joint_val)
    ever beat the plain 1-step search? Mirrors
    `notebooks/kstep_explore_search_labeling.ipynb`'s own batch-level check,
    condensed for a quick, non-plotting run."""
    from anytimeacquisition.pipelines.train_pfn import load_pfn_checkpoint
    from anytimeacquisition.search.explore import explore_search
    from anytimeacquisition.search.interesting_points import build_interesting_points
    from anytimeacquisition.utils.paths import CHECKPOINT_DIR

    torch.manual_seed(0)
    checkpoint_path = CHECKPOINT_DIR / "pfn_variable_xdim_smoke.pt"
    pfn, bar_dist, ckpt = load_pfn_checkpoint(checkpoint_path)
    print(f"loaded {checkpoint_path.name}, config={ckpt['config']}")

    x_dim = 1
    B = 10
    prior = BNNPrior(batch_size=B, x_dim=x_dim, seed=42)
    prior.reset()
    x_context, y_context, _, _ = prior.sample_episode(n_train=6, n_test=0)
    x_seed = torch.rand(B, x_dim)
    x_int, y_int_true = build_interesting_points(prior, n_sobol=24, n_random=16, n_basin_restarts=8, sobol_seed=0)

    x_star_1step, val_1step, has_signal_1step = explore_search(
        prior, pfn, bar_dist, x_context, y_context, x_int, y_int_true, x_seed,
        n_restarts=4, n_steps=40, lr=0.05,
    )
    x_star_k, val_k, has_signal_k, plan_star, joint_val = kstep_explore_search(
        prior, pfn, bar_dist, x_context, y_context, x_int, y_int_true, x_seed,
        k=3, n_restarts=4, n_steps=40, lr=0.05,
    )
    assert torch.equal(has_signal_1step, has_signal_k), "has_signal depends only on weights, must match"

    mask = has_signal_k
    gap = joint_val - val_k  # how much the joint score overstates x_star's real value
    lookahead_gain = val_k - val_1step  # >0 = k-step honestly beats 1-step

    n = int(mask.sum().item())
    print(f"{n}/{B} instances have signal")
    for b in range(B):
        if not mask[b]:
            continue
        print(
            f"  instance {b}: joint_val={joint_val[b].item():8.4f}  val_star(honest)={val_k[b].item():8.4f}  "
            f"gap={gap[b].item():+8.4f}  1-step val_star={val_1step[b].item():8.4f}  "
            f"lookahead_gain={lookahead_gain[b].item():+8.4f}"
        )
    if n:
        print(f"\nmean joint-vs-honest gap: {gap[mask].mean().item():+.4f} "
              f"(the amount trusting joint_val would overstate x_star by)")
        print(f"mean lookahead gain (k-step honest - 1-step honest): {lookahead_gain[mask].mean().item():+.4f}")
        print(f"fraction where k-step honestly beats 1-step: "
              f"{(lookahead_gain[mask] > 1e-6).float().mean().item():.2%}")
