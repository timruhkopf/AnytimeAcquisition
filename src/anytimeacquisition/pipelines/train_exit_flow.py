"""Flow-matching EXIT training pipeline -- the chunked-plan counterpart to
`pipelines/train_exit.py`, trains a `models.action_head_flow.FlowMatchingActionHead`
against the same privileged-search oracles via
`trainer.action_head_flow_trainer.ActionHeadFlowTrainer`. See that trainer's
own module docstring and `docs/MILESTONES.md`'s prioritized experiment 4 for
why this exists as a separate, deliberately narrower pipeline rather than a
mode switch on `train_exit.py`: the flow head's chunked, multi-step targets
(`trainer.exit_rollout.build_explore_chunk_buffer`/`build_exploit_chunk_buffer`)
and its rectified-flow loss are a different training surface than the simple
head's single-point BC loss, and this is architecture groundwork to be
compared against the simple head's own results, not a replacement decided
in advance.

Same two-entry-point shape as `pipelines/train_exit.py`:
`train_action_head_flow(...)` (plain function, scalar kwargs, no Hydra/
MLflow) and `main(cfg)` (the Hydra entry point, `configs/train_exit_flow.yaml`).
Run a smoke test via:
  uv run python -m anytimeacquisition.pipelines.train_exit_flow \\
    experiment=action_head_flow_explore_smoke allow_dirty=true
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
from anytimeacquisition.models.action_head import pfn_dims
from anytimeacquisition.models.action_head_flow import FlowMatchingActionHead
from anytimeacquisition.pipelines.train_pfn import load_pfn_checkpoint
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.trainer.action_head_flow_trainer import ActionHeadFlowTrainer
from anytimeacquisition.utils.flatten import flatten
from anytimeacquisition.utils.paths import CHECKPOINT_DIR

log = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "pfn_smoke_xdim1.pt"


def train_action_head_flow(
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
    chunk_len: int = 3,
    num_sample_steps: int = 10,
    action_head_d_model: int = 64,
    action_head_n_heads: int = 4,
    action_head_d_ff: int = 128,
    action_head_dropout: float = 0.0,
    checkpoint_out_path=None,
) -> dict:
    """Plain-function entry point -- loads a frozen PFN checkpoint, builds
    its own `BNNPrior`/`FlowMatchingActionHead`/`ActionHeadFlowTrainer` from
    scalar kwargs. No Hydra/MLflow; see `main()` for the Hydra pipeline."""
    torch.manual_seed(seed)
    pfn, bar_dist, ckpt = load_pfn_checkpoint(checkpoint_path)
    x_dim = ckpt["config"]["max_x_dim"]
    d_model, n_layers = pfn_dims(pfn)

    prior = BNNPrior(batch_size=batch_size, x_dim=x_dim, seed=seed, **(prior_kwargs or {}))
    action_head = FlowMatchingActionHead(
        pfn_d_model=d_model, pfn_n_layers=n_layers, x_dim=x_dim, chunk_len=chunk_len,
        d_model=action_head_d_model, n_heads=action_head_n_heads,
        d_ff=action_head_d_ff, dropout=action_head_dropout,
    )
    action_head_config = dict(
        pfn_d_model=d_model, pfn_n_layers=n_layers, x_dim=x_dim, chunk_len=chunk_len,
        d_model=action_head_d_model, n_heads=action_head_n_heads,
        d_ff=action_head_d_ff, dropout=action_head_dropout,
    )

    trainer = ActionHeadFlowTrainer(
        pfn=pfn, bar_dist=bar_dist, prior=prior, action_head=action_head,
        seed=seed, n_rollouts=n_rollouts, n_init=n_init, n_steps=n_steps, lr=lr, log_every=log_every,
        exploit_search_kwargs=exploit_search_kwargs, explore_search_kwargs=explore_search_kwargs,
        build_interesting_points_kwargs=build_interesting_points_kwargs,
        chunk_len=chunk_len, num_sample_steps=num_sample_steps,
        checkpoint_path=checkpoint_out_path, model_config=action_head_config,
    )
    return trainer.run()


def load_action_head_flow_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> tuple[FlowMatchingActionHead, dict]:
    """Symmetric to `pipelines.train_exit.load_action_head_checkpoint` --
    `ckpt["config"]` (`pfn_d_model`/`pfn_n_layers`/`x_dim`/`chunk_len`/
    `d_model`/`n_heads`/`d_ff`/`dropout`, see `ActionHeadFlowTrainer`'s own
    `model_config`) gets **-unpacked straight into
    `FlowMatchingActionHead(**ckpt["config"])`."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    action_head = FlowMatchingActionHead(**ckpt["config"]).to(device)
    action_head.load_state_dict(ckpt["model_state"])
    action_head.eval()
    return action_head, ckpt


@hydra.main(config_path="../../../configs", config_name="train_exit_flow", version_base=None)
def main(cfg: DictConfig) -> dict:
    """Hydra entry point. Select a named, reproducible config via
    `experiment=<name>` (see configs/experiment/), e.g.:
      uv run python -m anytimeacquisition.pipelines.train_exit_flow \\
        experiment=action_head_flow_explore_smoke
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
        # Plain kwargs, not `instantiate` -- action_head_flow: is inline
        # config (chunk_len/d_model/n_heads/d_ff/dropout), matching
        # configs/train_exit.yaml's action_head: convention.
        action_head_config = dict(
            pfn_d_model=d_model, pfn_n_layers=n_layers, x_dim=x_dim,
            chunk_len=cfg.action_head_flow.chunk_len,
            d_model=cfg.action_head_flow.d_model, n_heads=cfg.action_head_flow.n_heads,
            d_ff=cfg.action_head_flow.d_ff, dropout=cfg.action_head_flow.dropout,
        )
        action_head = FlowMatchingActionHead(**action_head_config)

        # Only auc_eval exists for the flow head today (see
        # configs/train_exit_flow.yaml's own comment on
        # action_head_validation: for why held_out_l1/blind_ablation/
        # explore_signal_rate aren't wired here).
        callbacks_by_name = instantiate(cfg.action_head_validation)
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
