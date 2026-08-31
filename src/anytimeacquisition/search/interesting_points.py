"""Builds the explore branch's fixed "interesting points" test-token set
(M5) — Sobol + random + GD-restart-found local/global minima, computed once
per `BNNPrior` instance **before** its rollout starts and never touched
again during that episode. Kept fixed deliberately (explicit user
direction, 2026-08-28): the explore branch's oracle search
(`search/explore.py`) needs a stable set of test tokens to weigh across the
whole trajectory, not a set that silently drifts as more context accumulates.

Where this differs from the design doc's original phrasing
(`archive/src/exit/PFN_ActionHead_ExpertIteration_Design.md` §2 point 5:
"drawn from the model's own current posterior... not from the privileged
ground-truth optimum") -- worth stating plainly rather than glossing over:
these points, and the GD restarts that find the basin-candidates among
them, use the BNN instance's own true, noise-free surface directly
(privileged information), not a trained PFN's posterior. That's a
deliberate scope choice, not an oversight: there is no trained
ActionHead/PFN-in-the-loop yet to draw "posterior-plausible" points from,
and generating the *candidate pool* from privileged information is still
confined to training-data generation (never deployment), same as the
exploit branch's search over the true surface. What must NOT be privileged
is the *weighting* decision at search time -- see `search/explore.py` for
why that stays keyed to the current realized incumbent, not future
information.
"""
import torch
from torch.quasirandom import SobolEngine

from anytimeacquisition.priors.bnn import BNNPrior


def find_basins(prior: BNNPrior, n_restarts: int = 16, n_steps: int = 50, lr: float = 0.05) -> torch.Tensor:
    """Multistart projected GD directly on `prior.evaluate(..., noise=False)`,
    same mechanism as `search.exploit.exploit_search`'s core loop but
    stripped of any context/incumbent dependency (there is no rollout yet
    when this runs) and returning *every* restart's final position, not
    just the best -- the point is a diverse set of local/global minima
    candidates, not a single answer.
    -> x [B, n_restarts, x_dim]
    """
    B, x_dim = prior.B, prior.d
    candidates = torch.rand(B, n_restarts, x_dim, requires_grad=True)
    opt = torch.optim.Adam([candidates], lr=lr)
    for _ in range(n_steps):
        y = prior.evaluate(candidates, noise=False)  # [B, n_restarts]
        opt.zero_grad()
        y.sum().backward()
        opt.step()
        with torch.no_grad():
            candidates.clamp_(0.0, 1.0)
    return candidates.detach()


def build_interesting_points(
    prior: BNNPrior, n_sobol: int = 32, n_random: int = 32, n_basin_restarts: int = 16,
    basin_search_kwargs: dict | None = None, sobol_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sobol (space-filling coverage) + uniform-random + GD-restart basin
    candidates (local/global minima), concatenated into one fixed
    per-instance test-point set, evaluated once against `prior`'s own
    noise-free surface. Call this exactly once per rollout episode, right
    after the instance is `reset()` and before any policy step runs (see
    `trainer.exit_rollout.rollout_episode`'s `build_interesting_points_kwargs`)
    -- the whole point is these stay constant for the rest of that episode.

    -> (x_int [B, n_sobol+n_random+n_basin_restarts, x_dim],
        y_int_true [B, n_sobol+n_random+n_basin_restarts]) -- the SAME Sobol
    draw is used for every instance in the batch (only `n_random`/the basin
    GD restarts differ per instance), since Sobol coverage doesn't depend on
    which instance it's covering.
    """
    basin_search_kwargs = basin_search_kwargs or {}
    B, x_dim = prior.B, prior.d

    if n_sobol > 0:
        sobol = SobolEngine(dimension=x_dim, scramble=True, seed=sobol_seed)
        x_sobol = sobol.draw(n_sobol).unsqueeze(0).expand(B, -1, -1)  # [B, n_sobol, x_dim]
    else:
        x_sobol = torch.empty(B, 0, x_dim)
    x_random = torch.rand(B, n_random, x_dim)
    x_basin = find_basins(prior, n_restarts=n_basin_restarts, **basin_search_kwargs)

    x_int = torch.cat([x_sobol, x_random, x_basin], dim=1)
    with torch.no_grad():
        y_int_true = prior.evaluate(x_int, noise=False)
    return x_int, y_int_true


if __name__ == "__main__":
    torch.manual_seed(0)
    prior = BNNPrior(batch_size=3, x_dim=2, seed=0)
    prior.reset()

    x_int, y_int_true = build_interesting_points(prior, n_sobol=20, n_random=20, n_basin_restarts=10)
    print("x_int shape:", tuple(x_int.shape), " y_int_true shape:", tuple(y_int_true.shape))
    print("y_int_true range per instance:", y_int_true.min(dim=1).values.tolist(), "..",
          y_int_true.max(dim=1).values.tolist())

    grid_res = 80
    lin = torch.linspace(0.0, 1.0, grid_res)
    grid = torch.stack(torch.meshgrid(lin, lin, indexing="ij"), dim=-1).reshape(1, -1, 2).expand(prior.B, -1, -1)
    with torch.no_grad():
        grid_best = prior.evaluate(grid, noise=False).min(dim=1).values
    print("dense-grid best y per instance (ground truth):", grid_best.tolist())
    print("interesting-points best y per instance:        ", y_int_true.min(dim=1).values.tolist())
    print("(basin restarts should get close to the grid's own best point)")
