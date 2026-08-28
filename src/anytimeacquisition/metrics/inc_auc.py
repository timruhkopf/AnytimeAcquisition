"""Log-incumbent AUC -- a realized-trajectory-only anytime-performance metric.

Ported/rebuilt from `archive/src/prototype/l2o_rlsf/model/rewards/
area_under_incumbent_curve.py` (raw-incumbent, reward-only) per M3: this
version works in log-space and is usable standalone for evaluation, not just
as an RL reward term. It only ever looks at y_0..y_t to compute the incumbent
at step t -- no privileged access to future steps or to the environment's
underlying parameters.

Two related but distinct quantities live here:
- `log_incumbent_auc`: the whole-trajectory metric (lower is better -- a
  search that reaches a good incumbent quickly keeps log(incumbent) low for
  most of the trajectory, so the sum stays low).
- `log_incumbent_stepwise_reward`: a *different* quantity, kept here only
  because it's derived from the same incumbent trajectory this module
  already computes. It telescopes to the total log-incumbent improvement
  (not the AUC above) and is what M5.5 will use as a dense per-step RL
  reward -- positive on steps that improve the incumbent, exactly zero
  otherwise.
"""
from pathlib import Path

import torch


def incumbent_trajectory(y: torch.Tensor, minimize: bool = True) -> torch.Tensor:
    """Realized running-best value at each step. y: [..., T] -> [..., T].

    incumbent_t depends only on y_0..y_t (no privileged access).
    """
    return torch.cummin(y, dim=-1).values if minimize else torch.cummax(y, dim=-1).values


def log_incumbent_auc(y: torch.Tensor, minimize: bool = True, eps: float = 1e-12) -> torch.Tensor:
    """Discrete area under the log-incumbent (step) curve. y: [..., T] -> [...].

    Lower is better. y must be > 0 (e.g. ECDF-normalized priors in (0, 1));
    `eps` floors the incumbent before `log` so a step landing exactly on 0
    doesn't produce -inf.
    """
    inc = incumbent_trajectory(y, minimize=minimize)
    return torch.log(inc.clamp_min(eps)).sum(dim=-1)


def log_incumbent_stepwise_reward(y: torch.Tensor, minimize: bool = True, eps: float = 1e-12) -> torch.Tensor:
    """Per-step reward r_t, t=1..T-1. y: [..., T] -> [..., T-1].

    r_t = log(incumbent_{t-1}) - log(incumbent_t) under minimize (sign
    flipped under maximize, so "positive" always means "improved" regardless
    of direction). Exactly 0 on steps that don't improve the incumbent,
    since incumbent_t == incumbent_{t-1} there. Sums (telescopes) to
    log(incumbent_0) - log(incumbent_{T-1}) -- the total log-incumbent
    improvement, distinct from `log_incumbent_auc` above.
    """
    log_inc = torch.log(incumbent_trajectory(y, minimize=minimize).clamp_min(eps))
    diff = log_inc[..., :-1] - log_inc[..., 1:]
    return diff if minimize else -diff


def plot_incumbent_curve(y: torch.Tensor, ax=None, minimize: bool = True):
    """Plot raw values vs. the incumbent step curve for one trajectory. y: [T].

    Port of the prototype's `AUICReward.plot_reward`.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    y_np = y.detach().cpu().numpy()
    inc_np = incumbent_trajectory(y, minimize=minimize).detach().cpu().numpy()
    steps = np.arange(len(y_np))

    ax.plot(steps, y_np, color="gray", alpha=0.3, label="raw y")
    ax.step(steps, inc_np, where="post", color="tab:blue", lw=2, label="incumbent")
    fill_base = inc_np.max() if minimize else inc_np.min()
    ax.fill_between(steps, inc_np, fill_base, step="post", alpha=0.1, color="tab:blue")
    ax.set_title("Incumbent progression")
    ax.set_xlabel("step")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")
    return ax


if __name__ == "__main__":
    torch.manual_seed(1)
    y = torch.rand(30) * (1 - torch.linspace(0, 0.9, 30)) + 0.01  # noisy-but-improving demo trajectory

    ax = plot_incumbent_curve(y)
    out_path = Path(__file__).parent / "_demo_plots" / "inc_auc_demo.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    print("saved", out_path)

    auc = log_incumbent_auc(y)
    r = log_incumbent_stepwise_reward(y)
    inc = incumbent_trajectory(y)
    print("log-incumbent AUC (lower is better):", auc.item())
    print("stepwise reward sum (telescoped):", r.sum().item())
    print("log(inc[0]) - log(inc[-1])       :", (torch.log(inc[0]) - torch.log(inc[-1])).item())
