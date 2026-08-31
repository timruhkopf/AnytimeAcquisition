"""ActionHead confidence-building toy problem — distill the frozen PFN's own
closed-form posterior into the ActionHead, *without* the privileged-search
oracle, environment/rollout, or EXIT loop Phase 5 eventually needs.

Why this exists: M5's exploit/explore branches assume the ActionHead's
cross-attention pathway can learn to hit privileged-search targets. Those
targets are expensive to produce (multistart GD, entropy-gradient search)
and this repo doesn't have the environment/rollout machinery to generate
them yet. Before investing in that, this checks the one assumption
everything else depends on: can gradient descent through this cross-
attention pathway learn to extract "where is the posterior best" from the
frozen PFN's *own* internal hidden states at all? The target here is cheap
and closed-form -- for a set of candidate query points, decode
`bar_dist.mean(pfn(...))` and take the argmin per context. No BNN
differentiable surface, no search of any kind: literally just reading the
PFN's own already-calibrated output head. If the ActionHead can't learn
this, it won't learn the harder EXIT targets either.

Three stages, each isolating a different failure mode:
  1. memorize -- one fixed context, many steps. Pure optimization sanity
     check (can the Beta head even fit a single known target) -- not yet
     informative about whether the PFN link carries anything.
  2. generalize -- fresh context every step (`prior.reset()` +
     `sample_episode`, same pattern as `trainer/pfn_trainer.py`), evaluated
     on held-out contexts never seen during training.
  3. blind ablation -- stage 2 repeated with `ActionHead.forward(...,
     blind=True)` (PFN hidden states zeroed, aux features untouched, see
     `models/action_head.py`), same steps/seed/architecture otherwise. This
     is the "not done this pass" item flagged in `docs/milestones/M4.md`. If
     blind performs comparably to real, the ActionHead is fitting on aux-
     token/dataset statistics, not the PFN cache, and cross-attention isn't
     earning its place.

Hydra + MLflow, same shape as `pipelines/train_pfn.py`: a plain function
(`run_distillation(...)`, scalar kwargs, no Hydra/MLflow -- what
`tests/test_action_head_posterior_distill.py` calls into via the lower-level
helpers) plus `main(cfg)`, the Hydra entry point
(`configs/action_head_posterior_distill.yaml`) wrapped with the same
provenance + MLflow-file-store convention every pipeline here uses. Config
values (checkpoint path, stage sizes, ActionHead capacity) live in that
top-level config and `configs/experiment/action_head_posterior_distill_bigscale.yaml`
rather than argparse flags, per repo convention -- see that experiment
file's own comment for why it's explicitly a throwaway, not a permanent
named config like `pfn_smoke_*`.

Aux features are held at *canonical* values (step_count=0, remaining_budget
=1.0, improvement_trend=0), except `incumbent_value`, which is set honestly
from the sampled context (`y_train.min()`) since it's context-derived, not a
nuisance variable. This toy problem tests only the cross-attention link, not
aux-token/budget conditioning or multi-step credit assignment -- that needs
the real EXIT loop (Phase 5), not this script.
"""
import logging
import os
from pathlib import Path
from typing import Callable

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import hydra
import mlflow
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from anytimeacquisition.deployment.provenance import record_provenance
from anytimeacquisition.models.action_head import ActionHead, beta_mode, pfn_dims
from anytimeacquisition.models.bar_distribution import BarDistribution
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.pipelines.train_pfn import load_pfn_checkpoint
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.utils.flatten import flatten
from anytimeacquisition.utils.paths import CHECKPOINT_DIR

log = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "pfn_smoke_xdim1.pt"


def posterior_argmin_targets(
    pfn: PFN, bar_dist: BarDistribution, x_train: torch.Tensor, y_train: torch.Tensor,
    n_candidates: int = 500,
) -> torch.Tensor:
    """Closed-form "greedy exploit" target per context: sample `n_candidates`
    uniform-random query points, decode the frozen PFN's own posterior mean
    at each (`bar_dist.mean`), return the argmin per batch element.
    -> [B, x_dim]. Random candidates rather than a grid so this works at any
    x_dim without the grid's combinatorial blowup."""
    B, _, x_dim = x_train.shape
    candidates = torch.rand(B, n_candidates, x_dim)
    with torch.no_grad():
        logits = pfn(x_train, y_train, candidates)  # [B, n_candidates, n_bins]
        means = bar_dist.mean(logits)  # [B, n_candidates]
    best_idx = means.argmin(dim=1)  # [B]
    return candidates[torch.arange(B), best_idx]


def canonical_aux_features(x_train: torch.Tensor, y_train: torch.Tensor) -> dict:
    """step_count/remaining_budget/improvement_trend fixed at canonical
    values (this toy problem isn't testing aux/budget conditioning);
    incumbent_value set honestly from the sampled context since it's
    context-derived, not a nuisance variable."""
    B = x_train.shape[0]
    return {
        "step_count": torch.zeros(B),
        "remaining_budget": torch.ones(B),
        "incumbent_value": y_train.min(dim=1).values,
        "improvement_trend": torch.zeros(B),
    }


def beta_nll_loss(alpha: torch.Tensor, beta: torch.Tensor, target_x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """-> [B], summed over x_dim (product-of-marginals, matches the policy
    head's per-dimension-independent Beta structure)."""
    target_x = target_x.clamp(eps, 1.0 - eps)
    return -torch.distributions.Beta(alpha, beta).log_prob(target_x).sum(dim=-1)


def run_stage(
    pfn: PFN, bar_dist: BarDistribution, action_head: ActionHead, prior: BNNPrior,
    n_steps: int, n_train: int, lr: float, blind: bool, seed: int, log_every: int,
    fixed_context: tuple[torch.Tensor, torch.Tensor] | None = None, n_candidates: int = 500,
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

        target = posterior_argmin_targets(pfn, bar_dist, x_train, y_train, n_candidates=n_candidates)
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
    n_contexts: int, n_train: int, blind: bool, seed: int, n_candidates: int = 500,
) -> float:
    """Held-out mean |beta_mode - target| across `n_contexts` fresh
    contexts, drawn with a seed disjoint from training."""
    torch.manual_seed(seed)
    total, count = 0.0, 0
    for _ in range(n_contexts):
        prior.reset()
        x_train, y_train, _, _ = prior.sample_episode(n_train=n_train, n_test=0)
        target = posterior_argmin_targets(pfn, bar_dist, x_train, y_train, n_candidates=n_candidates)
        aux = canonical_aux_features(x_train, y_train)
        out = action_head(pfn, x_train, y_train, aux, blind=blind)
        mode = beta_mode(out["alpha"], out["beta"])
        total += (mode - target).abs().sum().item()
        count += target.numel()
    return total / count


def run_distillation(
    pfn: PFN, bar_dist: BarDistribution, prior: BNNPrior, x_dim: int,
    n_train_context: int, memorize_steps: int, generalize_steps: int, eval_contexts: int,
    n_candidates: int, lr: float, seed: int, log_every: int,
    action_head_d_model: int, action_head_n_heads: int, action_head_d_ff: int, action_head_dropout: float,
    on_log: Callable[[int, dict], None] | None = None,
) -> dict:
    """The three-stage run itself, independent of Hydra/MLflow -- `main(cfg)`
    below unpacks config into these scalar kwargs. `on_log(step, metrics)` is
    called at `log_every` cadence during each of the two `generalize` stages
    (memorize's own loss curve is summarized, not streamed, since it's a
    pure optimization sanity check, not the informative measurement)."""
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
    print("\n[stage 1/3] memorize: one fixed context, check the Beta head can fit a known target")
    torch.manual_seed(seed)
    memorize_head = build_action_head()
    prior.reset()
    fixed_context = prior.sample_episode(n_train=n_train_context, n_test=0)[:2]
    memorize_losses = run_stage(
        pfn, bar_dist, memorize_head, prior, n_steps=memorize_steps, n_train=n_train_context,
        lr=lr, blind=False, seed=seed, log_every=log_every, fixed_context=fixed_context,
        n_candidates=n_candidates,
    )
    print(f"  memorize: loss {memorize_losses[0]:.4f} -> {memorize_losses[-1]:.4f}")

    # --- Stage 2: generalize (real) ----------------------------------------
    print("\n[stage 2/3] generalize: fresh context every step, real ActionHead")
    torch.manual_seed(seed)
    real_head = build_action_head()
    real_losses = run_stage(
        pfn, bar_dist, real_head, prior, n_steps=generalize_steps, n_train=n_train_context,
        lr=lr, blind=False, seed=seed, log_every=log_every, n_candidates=n_candidates,
        on_log=prefixed("generalize_real"),
    )
    real_eval_l1 = eval_mean_l1(
        pfn, bar_dist, real_head, prior, n_contexts=eval_contexts, n_train=n_train_context,
        blind=False, seed=seed + 10_000, n_candidates=n_candidates,
    )
    print(f"  generalize (real):  held-out mean |mode - target| = {real_eval_l1:.4f}")

    # --- Stage 3: blind ablation --------------------------------------------
    print("\n[stage 3/3] blind ablation: same setup, PFN hidden states zeroed")
    torch.manual_seed(seed)
    blind_head = build_action_head()
    blind_losses = run_stage(
        pfn, bar_dist, blind_head, prior, n_steps=generalize_steps, n_train=n_train_context,
        lr=lr, blind=True, seed=seed, log_every=log_every, n_candidates=n_candidates,
        on_log=prefixed("generalize_blind"),
    )
    blind_eval_l1 = eval_mean_l1(
        pfn, bar_dist, blind_head, prior, n_contexts=eval_contexts, n_train=n_train_context,
        blind=True, seed=seed + 10_000, n_candidates=n_candidates,
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

    return {
        "memorize_losses": memorize_losses,
        "real_losses": real_losses,
        "blind_losses": blind_losses,
        "memorize_final_loss": memorize_losses[-1],
        "real_eval_l1": real_eval_l1,
        "blind_eval_l1": blind_eval_l1,
        "real_vs_blind_ratio": real_eval_l1 / blind_eval_l1,
        "passed": passed,
    }


@hydra.main(config_path="../../../configs", config_name="action_head_posterior_distill", version_base=None)
def main(cfg: DictConfig) -> dict:
    """Hydra entry point. Select a named, reproducible config via
    `experiment=<name>` (see configs/experiment/), e.g.:
      uv run python -m anytimeacquisition.pipelines.action_head_posterior_distill \\
        experiment=action_head_posterior_distill_bigscale
    """
    overrides = HydraConfig.get().overrides.task
    provenance = record_provenance(list(overrides), allow_dirty=cfg.get("allow_dirty", False))

    mlflow.set_tracking_uri(cfg.callbacks.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.callbacks.mlflow.experiment_name)

    checkpoint_path = cfg.pfn_checkpoint.checkpoint_path
    pfn, bar_dist, ckpt = load_pfn_checkpoint(checkpoint_path)
    print(f"loaded PFN checkpoint: {Path(checkpoint_path).name}, config={ckpt['config']}")
    x_dim = ckpt["config"]["max_x_dim"]
    # priors.x_dim can't drift from this by construction (it interpolates
    # from pfn_checkpoint.max_x_dim), but the descriptor file itself could
    # still be stale/hand-edited-wrong relative to the actual .pt -- that's
    # the residual risk this validates (there's no way to auto-sync a
    # static config against a binary artifact).
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
        # Checkpoint lineage: the embedded value (train_pfn.py's main())
        # is ground truth for this specific file; fall back to the
        # descriptor's own hand-maintained copy, then "unknown" -- see
        # configs/pfn_checkpoint/smoke_xdim1.yaml for why both exist.
        mlflow.set_tags({
            "pfn_mlflow_run_id": ckpt.get("mlflow_run_id") or cfg.pfn_checkpoint.get("mlflow_run_id") or "unknown",
            "pfn_git_commit": ckpt.get("git_commit") or cfg.pfn_checkpoint.get("git_commit") or "unknown",
        })
        mlflow.log_params(flatten(OmegaConf.to_container(cfg, resolve=True)))
        # ckpt["config"] is a plain dict for checkpoints saved via
        # train_pfn()'s non-Hydra path, but an OmegaConf DictConfig for ones
        # saved via its Hydra path (train_pfn.py's main() doesn't always
        # convert before saving) -- dict(...) normalizes both so
        # flatten()'s isinstance(v, dict) check doesn't silently skip a
        # DictConfig and collapse it into one stringified param.
        mlflow.log_params(flatten({"pfn_checkpoint": dict(ckpt["config"])}))

        prior = instantiate(cfg.priors, seed=cfg.seed)

        result = run_distillation(
            pfn=pfn, bar_dist=bar_dist, prior=prior, x_dim=x_dim,
            n_train_context=cfg.n_train_context, memorize_steps=cfg.memorize_steps,
            generalize_steps=cfg.generalize_steps, eval_contexts=cfg.eval_contexts,
            n_candidates=cfg.n_candidates, lr=cfg.lr, seed=cfg.seed, log_every=cfg.log_every,
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
        log.info(
            "run complete: real_eval_l1=%.4f blind_eval_l1=%.4f passed=%s",
            result["real_eval_l1"], result["blind_eval_l1"], result["passed"],
        )
        return result


if __name__ == "__main__":
    main()
