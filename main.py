import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pathlib import Path

from src.utils.filelogger_json import BufferedDictLogger
from src.utils.seeding import SeededRandomContext


@hydra.main(config_path="configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # torch.set_printoptions(precision=4, sci_mode=False)

    file_logger = BufferedDictLogger(file_path=Path().cwd() / "log.jsonl", buffer_size=1)

    with SeededRandomContext(cfg.seed):

        env = instantiate(cfg.environment, device=device, logger=file_logger)
        model = instantiate(cfg.model, logger=file_logger)
        trainer = instantiate(cfg.trainer, model=model, env=env, logger=file_logger)

        trainer.train(cfg.num_epochs)

    file_logger.flush()

    if False:
        import matplotlib.pyplot as plt;
        file_logger.df['loss'].plot();
        plt.show()

    n_traces=3

    initial_condition = env.sample_initial_condition(B=n_traces)
    X = model.generate(
        env, B=n_traces, T=25,
        initial_condition=initial_condition
    )

    env.plot3d(traces=X)

    exploration_X = model.explore(
        env, B=n_traces, T=25,
        initial_condition=initial_condition
    )

    env.plot3d(traces=exploration_X)




    return None




if __name__ == '__main__':
    main()