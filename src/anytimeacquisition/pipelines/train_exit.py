"""EXIT training pipeline (M5, renamed from `action_head_imitation.py`
2026-09-02) -- trains the `ActionHead` against the privileged-search
oracles (`search/exploit.py`, `search/explore.py`) via
`trainer.action_head_imitation_trainer.ActionHeadImitationTrainer`,
DAgger-mixed by default (round-0 pure `trainer.exit_rollout.random_policy`,
phased into the ActionHead's own rollout behavior -- see that trainer's own
module docstring for the schedule).

One pipeline (not three): whether `cfg.trainer.exploit_search_kwargs`/
`explore_search_kwargs` are set (a dict) or `null` selects the
marginal-exploit, marginal-explore, or integrated run -- see
`configs/experiment/action_head_imitation_{exploit,explore,integrated}_smoke.yaml`
(no separate `branches` config value -- see
`trainer/action_head_imitation_trainer.py`'s own module docstring for why).
Same two-entry-point shape as `pipelines/train_pfn.py`:
`train_action_head_imitation(...)` (plain function, scalar kwargs, no
Hydra/MLflow) and `main(cfg)` (the Hydra entry point, `pfn_checkpoint`
config group + provenance + MLflow).

`callbacks/action_head_validation.py`'s four factory functions are
instantiated straight from `configs/train_exit.yaml`'s
`action_head_validation:` block via `_target_` (2026-09-02, previously
built explicitly here) -- each entry interpolates `x_dim`/`n_init`/
`n_steps` off `pfn_checkpoint`/`trainer` rather than needing them passed
programmatically, matching `configs/models/baselines/{ei,pi,es}.yaml`'s
own `_target_` convention.
"""
import logging
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import hydra
import mlflow
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from anytimeacquisition.deployment.provenance import record_provenance
from anytimeacquisition.models.action_head import ActionHead, pfn_dims
from anytimeacquisition.pipelines.train_pfn import load_pfn_checkpoint
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.trainer.action_head_imitation_trainer import ActionHeadImitationTrainer
from anytimeacquisition.utils.flatten import flatten
from anytimeacquisition.utils.paths import CHECKPOINT_DIR

log = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "pfn_smoke_xdim1.pt"


def train_action_head_imitation(
    checkpoint_path,
    seed: int = 0,
    n_rollouts: int = 200,
    n_init: int = 5,
    n_steps: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    log_every: int = 10,
    prior_kwargs: dict | None = None,
    exploit_search_kwargs: dict | None = None,
    explore_search_kwargs: dict | None = None,
    build_interesting_points_kwargs: dict | None = None,
    train_value_head: bool = False,
    action_head_d_model: int = 64,
    action_head_n_heads: int = 4,
    action_head_d_ff: int = 128,
    action_head_dropout: float = 0.0,
    checkpoint_out_path=None,
) -> dict:
    """Plain-function entry point -- loads a frozen PFN checkpoint, builds
    its own `BNNPrior`/`ActionHead`/`ActionHeadImitationTrainer` from scalar
    kwargs. No Hydra/MLflow; see `main()` for the Hydra pipeline."""
    torch.manual_seed(seed)
    pfn, bar_dist, ckpt = load_pfn_checkpoint(checkpoint_path)
    x_dim = ckpt["config"]["max_x_dim"]
    d_model, n_layers = pfn_dims(pfn)

    prior = BNNPrior(batch_size=batch_size, x_dim=x_dim, seed=seed, **(prior_kwargs or {}))
    action_head = ActionHead(
        pfn_d_model=d_model, pfn_n_layers=n_layers, x_dim=x_dim,
        d_model=action_head_d_model, n_heads=action_head_n_heads,
        d_ff=action_head_d_ff, dropout=action_head_dropout,
    )
    action_head_config = dict(
        pfn_d_model=d_model, pfn_n_layers=n_layers, x_dim=x_dim,
        d_model=action_head_d_model, n_heads=action_head_n_heads,
        d_ff=action_head_d_ff, dropout=action_head_dropout,
    )

    trainer = ActionHeadImitationTrainer(
        pfn=pfn, bar_dist=bar_dist, prior=prior, action_head=action_head,
        seed=seed, n_rollouts=n_rollouts, n_init=n_init, n_steps=n_steps, lr=lr, log_every=log_every,
        exploit_search_kwargs=exploit_search_kwargs, explore_search_kwargs=explore_search_kwargs,
        build_interesting_points_kwargs=build_interesting_points_kwargs, train_value_head=train_value_head,
        checkpoint_path=checkpoint_out_path, model_config=action_head_config,
    )
    return trainer.run()


def load_action_head_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> tuple[ActionHead, dict]:
    """Symmetric to `pipelines.train_pfn.load_pfn_checkpoint` -- `ckpt["config"]`
    (`pfn_d_model`/`pfn_n_layers`/`x_dim`/`d_model`/`n_heads`/`d_ff`/`dropout`,
    see `ActionHeadImitationTrainer`'s own `model_config`) gets **-unpacked
    straight into `ActionHead(**ckpt["config"])`. No old-format tolerance
    needed (unlike the PFN loader) -- this is a brand-new checkpoint format,
    every existing checkpoint already has the current shape."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    action_head = ActionHead(**ckpt["config"]).to(device)
    action_head.load_state_dict(ckpt["model_state"])
    action_head.eval()
    return action_head, ckpt


@hydra.main(config_path="../../../configs", config_name="train_exit", version_base=None)
def main(cfg: DictConfig) -> dict:
    """Hydra entry point. Select a named, reproducible config via
    `experiment=<name>` (see configs/experiment/), e.g.:
      uv run python -m anytimeacquisition.pipelines.train_exit \\
        experiment=action_head_imitation_integrated_smoke
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
    d_model, n_layers = pfn_dims(pfn)

    with mlflow.start_run():
        mlflow.set_tags(provenance.as_mlflow_tags())
        mlflow.set_tags({
            "pfn_mlflow_run_id": ckpt.get("mlflow_run_id") or cfg.pfn_checkpoint.get("mlflow_run_id") or "unknown",
            "pfn_git_commit": ckpt.get("git_commit") or cfg.pfn_checkpoint.get("git_commit") or "unknown",
        })
        mlflow.log_params(flatten(OmegaConf.to_container(cfg, resolve=True)))
        mlflow.log_params(flatten({"pfn_checkpoint": dict(ckpt["config"])}))

        prior = instantiate(cfg.priors, seed=cfg.seed)
        # Plain kwargs, not `instantiate` -- action_head_posterior_distill.yaml's
        # own `action_head:` block is inline config (d_model/n_heads/d_ff/
        # dropout), not a `_target_`-bearing Hydra group; matching that
        # convention rather than introducing a new one.
        action_head_config = dict(
            pfn_d_model=d_model, pfn_n_layers=n_layers, x_dim=x_dim,
            d_model=cfg.action_head.d_model, n_heads=cfg.action_head.n_heads,
            d_ff=cfg.action_head.d_ff, dropout=cfg.action_head.dropout,
        )
        action_head = ActionHead(**action_head_config)

        # Each entry under action_head_validation: carries its own
        # `_target_` (2026-09-02, previously built explicitly here --
        # x_dim/n_init/n_steps/build_interesting_points_kwargs are all
        # interpolated off pfn_checkpoint/trainer/the top-level config now,
        # see configs/train_exit.yaml). instantiate() on the whole block
        # eagerly calls each factory once, giving back {name: Callback}.
        callbacks_by_name = instantiate(cfg.action_head_validation)
        if cfg.trainer.explore_search_kwargs is None:
            # explore_signal_rate's own probe() always runs an uncapped
            # build_explore_buffer regardless of what the trainer trains
            # on -- skip it entirely when explore is off, not a branch-
            # taxonomy check, a compute-cost one (see that callback's own
            # docstring and docs/log/2026-09-01-explore-branch-beta-nll-uniform-collapse.md).
            callbacks_by_name.pop("explore_signal_rate", None)
        callbacks = list(callbacks_by_name.values())

        trainer = instantiate(
            cfg.trainer, pfn=pfn, bar_dist=bar_dist, prior=prior, action_head=action_head,
            seed=cfg.seed,
            model_config=action_head_config,
            on_log=lambda step, metrics: mlflow.log_metrics(metrics, step=step),
            extra_checkpoint_metadata={
                "mlflow_run_id": mlflow.active_run().info.run_id,
                "git_commit": provenance.commit,
                "pfn_mlflow_run_id": ckpt.get("mlflow_run_id") or "unknown",
                "pfn_git_commit": ckpt.get("git_commit") or "unknown",
            },
            callbacks=callbacks,
        )
        result = trainer.run()

        log.info("run complete, final metrics: %s", {k: v[-1] for k, v in result["history"].items() if k != "step"})
        return result["history"]


if __name__ == "__main__":
    main()
