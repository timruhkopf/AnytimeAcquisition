"""Causal-masked PFN training pipeline — a separate pretraining run from
`pipelines/train_pfn.py`'s bidirectional PFN, per `docs/ROADMAP.md` §7.4's
"worth a controlled comparison, not an a priori rejection." `causal=True`
(`models/pfn.py`) restricts train-train self-attention to lower-triangular
instead of bidirectional: train token `i` only attends to train tokens
`1..i`. This is a genuinely different model — its context representations
are no longer permutation-invariant (order now carries information; see
`docs/PROBLEM_SETTING.md` §PS.1/§PS.4) — trained from scratch here, not a
fine-tune of an existing bidirectional checkpoint.

**Why build this at all** (`docs/ROADMAP.md` §1.6/§7.4): the bidirectional
PFN's context representation is invalidated by any new observation, so
scoring a full BO trajectory recomputes the whole context from scratch at
every step (`O(B²)` token-work per episode — small in absolute terms, but
real, see `docs/PROBLEM_SETTING.md` §PS.4). A causally-masked model, by
construction, produces the *exact* representation each prefix length would
have produced on its own (`tests/test_pfn.py::test_causal_train_prefix_is_invariant_to_later_tokens`
verifies this directly) — so a real trajectory can extend a KV cache
incrementally instead of recomputing it, one new train token's own
self-attention plus whatever candidates are scored that step, not the
whole context again. The cost this trades away: exact exchangeability. The
acquisition-ordered context at BO decision time isn't an i.i.d. draw from
the input measure to begin with (§7.4's own point) — whether that makes the
lost exchangeability cheap or expensive in practice is exactly what a
controlled comparison against the bidirectional checkpoint should answer;
this pipeline produces the checkpoint that comparison needs, it doesn't
run the comparison itself.

**Not built here (scoped out deliberately):** the incremental-KV-cache
*inference* wrapper (the thing that would actually walk a trajectory
extending a cache instead of recomputing). That needs a trained causal
checkpoint to test against, which is what this pipeline produces first —
build the inference wrapper as its own next step, against a real
checkpoint, not speculatively here.

Same two entry points as `train_pfn.py`, by design (mirrors that pipeline
rather than diverging from it):
- `train_pfn_causal(...)` — plain function, no Hydra/MLflow.
- `main(cfg)` — the Hydra pipeline (`configs/train_pfn_causal.yaml`), own
  MLflow experiment (`anytimeacquisition-pfn-causal-pretrain`) so the two
  checkpoint lineages never mix in tracking. Run via:
    uv run python -m anytimeacquisition.pipelines.train_pfn_causal experiment=pfn_causal_smoke_xdim2
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

from anytimeacquisition.callbacks.dim_validation import build_dim_validation_callback
from anytimeacquisition.deployment.provenance import record_provenance
from anytimeacquisition.models.pfn import PFN
from anytimeacquisition.pipelines.train_pfn import load_pfn_checkpoint  # generic: reads causal from ckpt config
from anytimeacquisition.priors.bnn import BNNPrior
from anytimeacquisition.trainer.pfn_trainer import PFNTrainer
from anytimeacquisition.utils.flatten import flatten

log = logging.getLogger(__name__)


def train_pfn_causal(
    x_dim: int = 2,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 3,
    d_ff: int = 128,
    n_bins: int = 64,
    batch_size: int = 32,
    min_train: int = 3,
    max_train: int = 20,
    n_test: int = 10,
    n_steps: int = 500,
    lr: float = 1e-3,
    warmup_steps: int = 50,
    device: str = "cpu",
    seed: int = 0,
    prior_kwargs: dict | None = None,
    checkpoint_path: str | Path | None = None,
    log_every: int = 50,
    mixed_precision: bool = False,
) -> dict:
    """Plain-function entry point, mirrors `train_pfn.train_pfn` exactly
    except the model is built with `causal=True`. No Hydra/MLflow — see
    `main()` for the Hydra pipeline."""
    torch.manual_seed(seed)
    prior_kwargs = prior_kwargs or {}
    prior = BNNPrior(batch_size=batch_size, x_dim=x_dim, device=device, seed=seed, **prior_kwargs)
    model = PFN(
        max_x_dim=x_dim, d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, n_bins=n_bins, causal=True,
    ).to(device)
    model_config = dict(
        max_x_dim=x_dim, d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, n_bins=n_bins, causal=True,
    )

    trainer = PFNTrainer(
        prior=prior, model=model, seed=seed, n_steps=n_steps, min_train=min_train,
        max_train=max_train, n_test=n_test, lr=lr, warmup_steps=warmup_steps, log_every=log_every,
        checkpoint_path=checkpoint_path, model_config=model_config, mixed_precision=mixed_precision,
    )
    return trainer.run()


@hydra.main(config_path="../../../configs", config_name="train_pfn_causal", version_base=None)
def main(cfg: DictConfig) -> dict:
    """Hydra entry point. Select a named, reproducible config via
    `experiment=<name>` (see configs/experiment/pfn_causal_*.yaml), e.g.:
      uv run python -m anytimeacquisition.pipelines.train_pfn_causal experiment=pfn_causal_smoke_xdim2
    """
    overrides = HydraConfig.get().overrides.task
    provenance = record_provenance(list(overrides), allow_dirty=cfg.get("allow_dirty", False))

    mlflow.set_tracking_uri(cfg.callbacks.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.callbacks.mlflow.experiment_name)

    with mlflow.start_run():
        mlflow.set_tags(provenance.as_mlflow_tags())
        mlflow.log_params(flatten(OmegaConf.to_container(cfg, resolve=True)))

        prior = instantiate(cfg.priors, seed=cfg.seed, device=cfg.device)
        model = instantiate(cfg.models.surrogates).to(cfg.device)
        model_config = OmegaConf.to_container(cfg.models.surrogates, resolve=True)
        model_config.pop("_target_")

        # Same opt-in per-dimension validation as train_pfn.py -- see that
        # module's own main() for the full rationale.
        validate_dims = cfg.get("validate_dims", None)
        callbacks = None
        if validate_dims:
            prior_kwargs = OmegaConf.to_container(cfg.priors, resolve=True)
            for key in (
                "_target_", "x_dim", "variable_dim_min", "batch_size", "seed",
                "ecdf_n_samples", "ecdf_n_draws", "ecdf_samples_per_draw", "cache_dir",
            ):
                prior_kwargs.pop(key, None)
            callbacks = [build_dim_validation_callback(
                dims=list(validate_dims), max_x_dim=cfg.priors.x_dim,
                ecdf_sorted=prior.ecdf_sorted, prior_kwargs=prior_kwargs, seed=cfg.seed,
            )]

        trainer = instantiate(
            cfg.trainer, prior=prior, model=model, seed=cfg.seed, model_config=model_config,
            on_log=lambda step, metrics: mlflow.log_metrics(metrics, step=step),
            extra_checkpoint_metadata={
                "mlflow_run_id": mlflow.active_run().info.run_id,
                "git_commit": provenance.commit,
            },
            callbacks=callbacks,
        )
        result = trainer.run()

        log.info("run complete, final metrics: %s", {k: v[-1] for k, v in result["history"].items() if k != "step"})
        return result["history"]


if __name__ == "__main__":
    main()
