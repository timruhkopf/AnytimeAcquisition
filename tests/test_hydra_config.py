from hydra import compose, initialize
from hydra.utils import instantiate


def test_default_config_composes_and_instantiates():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")

        benchmark = instantiate(cfg.benchmarks)
        prior = instantiate(cfg.priors)
        surrogate = instantiate(cfg.models.surrogates)
        trainer = instantiate(
            cfg.trainer, benchmark=benchmark, prior=prior, surrogate=surrogate
        )

        result = trainer.run()

        assert result["n_steps"] == cfg.trainer.n_steps
        assert "best_y" in result
