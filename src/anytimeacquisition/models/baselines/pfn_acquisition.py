"""Expected Improvement computed directly off the frozen PFN's own PPD --
not a GP (see `gp_acquisition.py` for that baseline; this is a *different*
surrogate, both explicitly named in `docs/ROADMAP.md` Phase 5: "a classical
acquisition function (EI/UCB) run on the same frozen PFN's own PPD").

The closed-form EI/PI formulas themselves live on `BarDistribution`
(`.ei()`/`.pi()`) as of 2026-09-08 -- ported there from PFNs4BO's own
`archive/src/utils/bar_distribution.py::BarDistribution.ei()`/`.pi()`,
algebraically mirrored for this project's minimize convention (see
`bar_distribution.py`'s own docstrings for the exact mirroring and the
Monte Carlo cross-check). This module now only holds `pfn_ei_argmax`'s
dense-grid search.

`pfn_ei_argmax` is deliberately **not** an optimizer: it evaluates a dense
`torch.linspace` grid in one batched PFN forward call and takes the argmax.
No multistart gradient descent, no local-optimum risk -- the only
approximation is grid resolution, which is directly checkable (`n_grid`
higher vs lower), unlike an optimizer's convergence. This was a deliberate
choice for the argmax-finding diagnostic this module was built for
(`docs/log/2026-09-02-actionhead-search-depth-design-options.md`,
`pipelines/action_head_ei_diagnostic.py`): reusing `search/explore.py`'s
multistart-GD machinery here would let the oracle's own optimizer be a
second, entangled source of error on top of whatever's being diagnosed --
exactly what that diagnostic exists to avoid. x_dim=1 only, for the same
reason (a dense grid stops being cheap/exhaustive in higher dimensions;
revisit with a proper search once x_dim=1 results justify it).
"""
import torch

from anytimeacquisition.models.bar_distribution import BarDistribution
from anytimeacquisition.models.pfn import PFN


def pfn_ei_argmax(
    pfn: PFN, bar_dist: BarDistribution, x_train: torch.Tensor, y_train: torch.Tensor, n_grid: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x_train: [B,Ntr,1]  y_train: [B,Ntr] (x_dim=1 only, asserted). Dense
    `torch.linspace(0,1,n_grid)` grid, shared across the batch (mirrors
    `search.interesting_points.build_interesting_points`'s shared-Sobol-
    draw convention), ONE batched `pfn(...)` forward call -- train-side
    self-attention runs once regardless of grid density, since test tokens
    never influence train tokens (`models/pfn.py`'s `PFNBlock.forward`),
    so this is already as cheap as it can be without further caching.
    -> (x_star [B,1], grid [n_grid,1], ei_grid [B,n_grid])."""
    assert x_train.shape[-1] == 1, f"pfn_ei_argmax is 1D-only, got x_dim={x_train.shape[-1]}"
    B = x_train.shape[0]
    grid = torch.linspace(0.0, 1.0, n_grid, device=x_train.device).view(n_grid, 1)
    grid_batched = grid.unsqueeze(0).expand(B, -1, -1)
    with torch.no_grad():
        logits = pfn(x_train, y_train, grid_batched)  # [B, n_grid, n_bins]
    incumbent = y_train.min(dim=1).values  # [B]
    ei_grid = bar_dist.ei(logits, incumbent.view(B, 1))  # [B, n_grid]
    best_idx = ei_grid.argmax(dim=1)
    x_star = grid_batched[torch.arange(B), best_idx]
    return x_star, grid, ei_grid


def pfn_acquisition_policy(
    x_context: torch.Tensor, y_context: torch.Tensor, x_dim: int,
    pfn: PFN, bar_dist: BarDistribution, n_grid: int = 1000,
) -> torch.Tensor:
    """Same signature/contract as `gp_acquisition_policy`/`random_policy` --
    drop-in `policy_fn` for `rollout_episode` (bind `pfn`/`bar_dist` via
    `functools.partial`, matching the `__main__` demo below and
    `gp_acquisition.py`'s own usage pattern). x_dim must be 1."""
    assert x_dim == 1, f"pfn_acquisition_policy is 1D-only, got x_dim={x_dim}"
    x_star, _, _ = pfn_ei_argmax(pfn, bar_dist, x_context, y_context, n_grid=n_grid)
    return x_star


if __name__ == "__main__":
    """Load pfn_smoke_xdim1.pt, sample a context, plot the PFN's posterior
    mean curve + the EI curve + the argmax marker -- no Hydra, no training,
    fast. Demo for this module specifically; the full training+eval flow
    lives in pipelines/action_head_ei_diagnostic.py."""
    import matplotlib.pyplot as plt

    from anytimeacquisition.pipelines.train_pfn import load_pfn_checkpoint
    from anytimeacquisition.priors.bnn import BNNPrior
    from anytimeacquisition.utils.paths import CHECKPOINT_DIR

    checkpoint_path = CHECKPOINT_DIR / "pfn_smoke_xdim1.pt"
    if not checkpoint_path.exists():
        raise SystemExit(
            f"No checkpoint at {checkpoint_path} -- train one first:\n"
            "  uv run python -m anytimeacquisition.pipelines.train_pfn "
            "experiment=pfn_smoke_xdim1 allow_dirty=true"
        )
    pfn, bar_dist, ckpt = load_pfn_checkpoint(checkpoint_path)
    print(f"loaded PFN checkpoint: {checkpoint_path.name}, config={ckpt['config']}")

    torch.manual_seed(0)
    prior = BNNPrior(batch_size=1, x_dim=1, seed=0)
    prior.reset()
    x_train, y_train, _, _ = prior.sample_episode(n_train=8, n_test=0)

    x_star, grid, ei_grid = pfn_ei_argmax(pfn, bar_dist, x_train, y_train, n_grid=1000)
    with torch.no_grad():
        logits_grid = pfn(x_train, y_train, grid.unsqueeze(0))
        mean_grid = bar_dist.mean(logits_grid)[0]
    incumbent = y_train.min(dim=1).values.item()
    print(f"incumbent: {incumbent:.4f}  x_star (EI argmax): {x_star.item():.4f}  "
          f"EI(x_star): {ei_grid.max().item():.4f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    ax1.plot(grid.squeeze(-1), mean_grid, label="PFN posterior mean")
    ax1.scatter(x_train[0, :, 0], y_train[0], color="black", marker="x", label="context")
    ax1.axhline(incumbent, color="gray", linestyle="--", linewidth=1, label="incumbent")
    ax1.set_ylabel("y (minimize)")
    ax1.legend()
    ax2.plot(grid.squeeze(-1), ei_grid[0], color="tab:orange", label="EI(x)")
    ax2.axvline(x_star.item(), color="tab:red", linestyle="--", label="EI argmax")
    ax2.set_xlabel("x")
    ax2.set_ylabel("EI")
    ax2.legend()
    fig.tight_layout()
    out_path = CHECKPOINT_DIR.parent / "outputs" / "pfn_acquisition_demo.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"saved plot to {out_path}")
