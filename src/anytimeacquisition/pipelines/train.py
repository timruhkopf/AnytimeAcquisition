import logging
import os

# MLflow 3.x deprecated the plain filesystem tracking store in favor of DB
# backends. We deliberately stay on the file store (no Postgres/SQLite server
# to coordinate across parallel SLURM jobs), which requires opting back in.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import hydra
import mlflow
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from anytimeacquisition.deployment.provenance import record_provenance
from anytimeacquisition.utils.flatten import flatten

log = logging.getLogger(__name__)


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> dict:
    overrides = HydraConfig.get().overrides.task
    provenance = record_provenance(list(overrides), allow_dirty=cfg.get("allow_dirty", False))

    mlflow.set_tracking_uri(cfg.callbacks.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.callbacks.mlflow.experiment_name)

    with mlflow.start_run():
        mlflow.set_tags(provenance.as_mlflow_tags())
        mlflow.log_params(flatten(OmegaConf.to_container(cfg, resolve=True)))

        benchmark = instantiate(cfg.benchmarks)
        prior = instantiate(cfg.priors)
        surrogate = instantiate(cfg.models.surrogates)
        trainer = instantiate(cfg.trainer, benchmark=benchmark, prior=prior, surrogate=surrogate)

        result = trainer.run()
        mlflow.log_metrics({k: v for k, v in result.items() if isinstance(v, (int, float))})
        log.info("run complete: %s", result)
        return result


if __name__ == "__main__":
    main()
