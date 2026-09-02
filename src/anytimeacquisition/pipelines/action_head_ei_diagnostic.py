"""Isolates whether PFN+ActionHead's single-shot cross-attention readout can
find the argmax of a KNOWN, cheap, closed-form acquisition function (EI
computed directly off the frozen PFN's own PPD,
`models/baselines/pfn_acquisition.py::pfn_ei_argmax`) -- decoupling "can
this architecture do argmax-finding at all" from "is the privileged-search
oracle's target even learnable", the two questions the full EXIT pipeline
(Phase 5) currently entangles into one training signal. Same "T-maze
isolates credit assignment from full RL" logic, applied here to
argmax-finding instead. See
`docs/log/2026-09-02-actionhead-search-depth-design-options.md` for the
design options (K-candidate tokens, recursive refinement, flow matching)
this diagnostic is meant to be run against once/if the current single-token
architecture underperforms here.

x_dim=1 only, deliberately (`pfn_ei_argmax`'s dense-grid oracle is 1D-only,
see that module's docstring for why a search/optimizer was deliberately
NOT used instead) -- this also makes the whole thing directly
visualizable: ground-truth EI curve + argmax + the trained policy's own
prediction, overlaid (`plot_ei_diagnostic` below). Related to, though
narrower than, `docs/milestones/M5.md`'s still-open "1D interpretability
diagnostic" checklist item (which additionally wants the entropy curve,
UCB/PI, and the ActionHead's own policy density -- none of those are built
here, this is scoped to EI + argmax only, per explicit user direction).

Three stages, the same failure-mode-isolating ladder
`pipelines/action_head_posterior_distill.py` established for its own
(easier, pure posterior-mean-argmin) target -- built independently here,
not imported from that file, per explicit user direction to keep this a
fresh, separate pipeline rather than extend the orphaned one:
  1. memorize -- one fixed context, many steps. Pure optimization sanity
     check (can the Beta head even fit a single known EI-argmax target).
  2. generalize -- fresh context every step, held-out eval.
  3. blind ablation -- same setup, PFN hidden states zeroed
     (`ActionHead.forward(..., blind=True)`). If blind matches real, the
     ActionHead isn't using the PFN link for this target either.

This target is genuinely harder than `action_head_posterior_distill.py`'s
posterior-mean-argmin: EI depends on both the posterior mean *and*
variance *and* the current incumbent, not just "where is the mean lowest",
so a pass here is stronger evidence the cross-attention pathway can
extract more than a single summary statistic.
"""
import logging
import os
from pathlib import Path
from typing import Callable

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import hydra
import matplotlib.pyplot as plt
import mlflow
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from anytimeacquisition.deployment.provenance import record_provenance
from anytimeacquisition.models.action_head import ActionHead, beta_mode, pfn_dims
from anytimeacquisition.models.bar_distribution import BarDistribution
from anytimeacquisition.models.baselines.pfn_acquisition import pfn_ei_argmax
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.pipelines.train_pfn import load_pfn_checkpoint
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.utils.flatten import flatten
from anytimeacquisition.utils.paths import CHECKPOINT_DIR

log = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "pfn_smoke_xdim1.pt"


def canonical_aux_features(x_train: torch.Tensor, y_train: torch.Tensor) -> dict:
    """step_count/remaining_budget/improvement_trend fixed at canonical
    values (this diagnostic isn't testing aux/budget conditioning);
    incumbent_value set honestly from the sampled context since it's
    context-derived, not a nuisance variable -- same convention
    `action_head_posterior_distill.py::canonical_aux_features` uses."""
    B = x_train.shape[0]
    return {
        "step_count": torch.zeros(B),
        "remaining_budget": torch.ones(B),
        "incumbent_value": y_train.min(dim=1).values,
        "improvement_trend": torch.zeros(B),
    }


def beta_nll_loss(alpha: torch.Tensor, beta: torch.Tensor, target_x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """-> [B], summed over x_dim (trivial here since x_dim=1, kept as a sum
    for the same product-of-marginals shape every other Beta-NLL loss in
    this repo uses)."""
    target_x = target_x.clamp(eps, 1.0 - eps)
    return -torch.distributions.Beta(alpha, beta).log_prob(target_x).sum(dim=-1)


def run_stage(
    pfn: PFN, bar_dist: BarDistribution, action_head: ActionHead, prior: BNNPrior,
    n_steps: int, n_train: int, lr: float, blind: bool, seed: int, log_every: int,
    fixed_context: tuple[torch.Tensor, torch.Tensor] | None = None, n_grid: int = 1000,
    on_log: Callable[[int, dict], None] | None = None,
) -> list[float]:
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(action_head.parameters(), lr=lr)
    losses = []
    for step in range(n_steps):
        if fixed_context is not None:
            x_train, y_train = fixed_context
        else:
            prior.reset()
            x_train, y_train, _, _ = prior.sample_episode(n_train=n_train, n_test=0)

        target, _, _ = pfn_ei_argmax(pfn, bar_dist, x_train, y_train, n_grid=n_grid)
        aux = canonical_aux_features(x_train, y_train)
        out = action_head(pfn, x_train, y_train, aux, blind=blind)
        loss = beta_nll_loss(out["alpha"], out["beta"], target).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % log_every == 0 or step == n_steps - 1:
            print(f"    step {step:4d}  beta_nll {loss.item():8.4f}")
            if on_log is not None:
                on_log(step, {"beta_nll": loss.item()})
    return losses


@torch.no_grad()
def eval_mean_l1(
    pfn: PFN, bar_dist: BarDistribution, action_head: ActionHead, prior: BNNPrior,
    n_contexts: int, n_train: int, blind: bool, seed: int, n_grid: int = 1000,
) -> float:
    """Held-out mean |beta_mode - target| across `n_contexts` fresh
    contexts, drawn with a seed disjoint from training."""
    torch.manual_seed(seed)
    total, count = 0.0, 0
    for _ in range(n_contexts):
        prior.reset()
        x_train, y_train, _, _ = prior.sample_episode(n_train=n_train, n_test=0)
        target, _, _ = pfn_ei_argmax(pfn, bar_dist, x_train, y_train, n_grid=n_grid)
        aux = canonical_aux_features(x_train, y_train)
        out = action_head(pfn, x_train, y_train, aux, blind=blind)
        mode = beta_mode(out["alpha"], out["beta"])
        total += (mode - target).abs().sum().item()
        count += target.numel()
    return total / count


def plot_ei_diagnostic(
    pfn: PFN, bar_dist: BarDistribution, action_head: ActionHead, prior: BNNPrior,
    n_contexts: int = 4, n_train: int = 10, n_grid: int = 1000, seed: int = 12345,
):
    """A small grid of held-out-context subplots, each showing the
    ground-truth EI curve, its argmax, and the trained ActionHead's own
    `beta_mode` prediction overlaid -- the direct visual check "did the
    policy converge near the true argmax", not just an aggregate L1
    number. -> matplotlib Figure (caller saves/logs it)."""
    torch.manual_seed(seed)
    fig, axes = plt.subplots(1, n_contexts, figsize=(4 * n_contexts, 3.5), sharey=True)
    if n_contexts == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        prior.reset()
        x_train, y_train, _, _ = prior.sample_episode(n_train=n_train, n_test=0)
        x_star, grid, ei_grid = pfn_ei_argmax(pfn, bar_dist, x_train, y_train, n_grid=n_grid)
        aux = canonical_aux_features(x_train, y_train)
        with torch.no_grad():
            out = action_head(pfn, x_train, y_train, aux, blind=False)
            pred = beta_mode(out["alpha"], out["beta"])

        ax.plot(grid.squeeze(-1), ei_grid[0], color="tab:orange", label="EI(x)")
        ax.axvline(x_star[0].item(), color="tab:red", linestyle="--", label="EI argmax")
        ax.axvline(pred[0, 0].item(), color="tab:blue", linestyle=":", label="ActionHead")
        ax.scatter(x_train[0, :, 0], torch.zeros_like(x_train[0, :, 0]), color="black", marker="x", s=15)
        ax.set_title(f"context {i}")
        ax.set_xlabel("x")
        if i == 0:
            ax.set_ylabel("EI")
            ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def run_diagnostic(
    pfn: PFN, bar_dist: BarDistribution, prior: BNNPrior, x_dim: int,
    n_train_context: int, memorize_steps: int, generalize_steps: int, eval_contexts: int,
    n_grid: int, lr: float, seed: int, log_every: int,
    action_head_d_model: int, action_head_n_heads: int, action_head_d_ff: int, action_head_dropout: float,
    on_log: Callable[[int, dict], None] | None = None,
) -> dict:
    """The three-stage run itself, independent of Hydra/MLflow -- `main(cfg)`
    below unpacks config into these scalar kwargs, same shape as
    `action_head_posterior_distill.py::run_distillation`."""
    assert x_dim == 1, f"action_head_ei_diagnostic is 1D-only, got x_dim={x_dim}"
    d_model, n_layers = pfn_dims(pfn)

    def build_action_head() -> ActionHead:
        return ActionHead(
            pfn_d_model=d_model, pfn_n_layers=n_layers, x_dim=x_dim,
            d_model=action_head_d_model, n_heads=action_head_n_heads,
            d_ff=action_head_d_ff, dropout=action_head_dropout,
        )

    def prefixed(prefix: str) -> Callable[[int, dict], None] | None:
        if on_log is None:
            return None
        return lambda step, metrics: on_log(step, {f"{prefix}/{k}": v for k, v in metrics.items()})

    # --- Stage 1: memorize -------------------------------------------------
    print("\n[stage 1/3] memorize: one fixed context, check the Beta head can fit a known EI-argmax target")
    torch.manual_seed(seed)
    memorize_head = build_action_head()
    prior.reset()
    fixed_context = prior.sample_episode(n_train=n_train_context, n_test=0)[:2]
    memorize_losses = run_stage(
        pfn, bar_dist, memorize_head, prior, n_steps=memorize_steps, n_train=n_train_context,
        lr=lr, blind=False, seed=seed, log_every=log_every, fixed_context=fixed_context, n_grid=n_grid,
    )
    print(f"  memorize: loss {memorize_losses[0]:.4f} -> {memorize_losses[-1]:.4f}")

    # --- Stage 2: generalize (real) ----------------------------------------
    print("\n[stage 2/3] generalize: fresh context every step, real ActionHead")
    torch.manual_seed(seed)
    real_head = build_action_head()
    real_losses = run_stage(
        pfn, bar_dist, real_head, prior, n_steps=generalize_steps, n_train=n_train_context,
        lr=lr, blind=False, seed=seed, log_every=log_every, n_grid=n_grid,
        on_log=prefixed("generalize_real"),
    )
    real_eval_l1 = eval_mean_l1(
        pfn, bar_dist, real_head, prior, n_contexts=eval_contexts, n_train=n_train_context,
        blind=False, seed=seed + 10_000, n_grid=n_grid,
    )
    print(f"  generalize (real):  held-out mean |mode - target| = {real_eval_l1:.4f}")

    # --- Stage 3: blind ablation --------------------------------------------
    print("\n[stage 3/3] blind ablation: same setup, PFN hidden states zeroed")
    torch.manual_seed(seed)
    blind_head = build_action_head()
    blind_losses = run_stage(
        pfn, bar_dist, blind_head, prior, n_steps=generalize_steps, n_train=n_train_context,
        lr=lr, blind=True, seed=seed, log_every=log_every, n_grid=n_grid,
        on_log=prefixed("generalize_blind"),
    )
    blind_eval_l1 = eval_mean_l1(
        pfn, bar_dist, blind_head, prior, n_contexts=eval_contexts, n_train=n_train_context,
        blind=True, seed=seed + 10_000, n_grid=n_grid,
    )
    print(f"  generalize (blind): held-out mean |mode - target| = {blind_eval_l1:.4f}")

    print("\n=== summary ===")
    print(f"memorize final loss:            {memorize_losses[-1]:.4f}")
    print(f"generalize (real)  held-out L1: {real_eval_l1:.4f}")
    print(f"generalize (blind) held-out L1: {blind_eval_l1:.4f}")
    passed = real_eval_l1 < 0.8 * blind_eval_l1
    verdict = "PASS -- real clearly beats blind" if passed else \
        "INCONCLUSIVE/FAIL -- real does not clearly beat blind, cross-attention link may not be carrying signal"
    print(verdict)

    figure = plot_ei_diagnostic(pfn, bar_dist, real_head, prior, n_train=n_train_context, n_grid=n_grid)

    return {
        "memorize_losses": memorize_losses,
        "real_losses": real_losses,
        "blind_losses": blind_losses,
        "memorize_final_loss": memorize_losses[-1],
        "real_eval_l1": real_eval_l1,
        "blind_eval_l1": blind_eval_l1,
        "real_vs_blind_ratio": real_eval_l1 / blind_eval_l1,
        "passed": passed,
        "figure": figure,
    }


@hydra.main(config_path="../../../configs", config_name="action_head_ei_diagnostic", version_base=None)
def main(cfg: DictConfig) -> dict:
    """Hydra entry point.
      uv run python -m anytimeacquisition.pipelines.action_head_ei_diagnostic \\
        allow_dirty=true
    """
    overrides = HydraConfig.get().overrides.task
    provenance = record_provenance(list(overrides), allow_dirty=cfg.get("allow_dirty", False))

    mlflow.set_tracking_uri(cfg.callbacks.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.callbacks.mlflow.experiment_name)

    checkpoint_path = cfg.pfn_checkpoint.checkpoint_path
    pfn, bar_dist, ckpt = load_pfn_checkpoint(checkpoint_path)
    print(f"loaded PFN checkpoint: {Path(checkpoint_path).name}, config={ckpt['config']}")
    x_dim = ckpt["config"]["max_x_dim"]
    declared = {
        k: v for k, v in OmegaConf.to_container(cfg.pfn_checkpoint, resolve=True).items()
        if k not in ("checkpoint_path", "mlflow_run_id", "git_commit")
    }
    if declared != dict(ckpt["config"]):
        raise ValueError(
            f"configs/pfn_checkpoint descriptor {declared} does not match the checkpoint's own "
            f"config {dict(ckpt['config'])} ({checkpoint_path}) -- the descriptor is stale, update "
            "it to match the actual .pt file."
        )

    with mlflow.start_run():
        mlflow.set_tags(provenance.as_mlflow_tags())
        mlflow.set_tags({
            "pfn_mlflow_run_id": ckpt.get("mlflow_run_id") or cfg.pfn_checkpoint.get("mlflow_run_id") or "unknown",
            "pfn_git_commit": ckpt.get("git_commit") or cfg.pfn_checkpoint.get("git_commit") or "unknown",
        })
        mlflow.log_params(flatten(OmegaConf.to_container(cfg, resolve=True)))
        mlflow.log_params(flatten({"pfn_checkpoint": dict(ckpt["config"])}))

        prior = instantiate(cfg.priors, seed=cfg.seed)

        result = run_diagnostic(
            pfn=pfn, bar_dist=bar_dist, prior=prior, x_dim=x_dim,
            n_train_context=cfg.n_train_context, memorize_steps=cfg.memorize_steps,
            generalize_steps=cfg.generalize_steps, eval_contexts=cfg.eval_contexts,
            n_grid=cfg.n_grid, lr=cfg.lr, seed=cfg.seed, log_every=cfg.log_every,
            action_head_d_model=cfg.action_head.d_model, action_head_n_heads=cfg.action_head.n_heads,
            action_head_d_ff=cfg.action_head.d_ff, action_head_dropout=cfg.action_head.dropout,
            on_log=lambda step, metrics: mlflow.log_metrics(metrics, step=step),
        )

        mlflow.log_metrics({
            "memorize_final_loss": result["memorize_final_loss"],
            "real_eval_l1": result["real_eval_l1"],
            "blind_eval_l1": result["blind_eval_l1"],
            "real_vs_blind_ratio": result["real_vs_blind_ratio"],
            "passed": float(result["passed"]),
        })
        mlflow.log_figure(result["figure"], "ei_diagnostic.png")
        plt.close(result["figure"])
        log.info(
            "run complete: real_eval_l1=%.4f blind_eval_l1=%.4f passed=%s",
            result["real_eval_l1"], result["blind_eval_l1"], result["passed"],
        )
        return {k: v for k, v in result.items() if k != "figure"}


if __name__ == "__main__":
    main()
