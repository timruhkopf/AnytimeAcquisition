from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import logging



@hydra.main(version_base='1.1', config_path="../../configs", config_name="base")
def main(cfg: DictConfig) -> None:

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO) # TODO make this yaml configurable

    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # set seed for reproducibility
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg.seed)

    env = instantiate(cfg.environment.env_class, device=device)
    policy = instantiate(cfg.model.policy_class)


    trainer = instantiate(
        cfg.trainer.trainer_class,
        device=device,
        env=env,
        policy=policy,
    )

    # dictconfig cannot be passed directly; neither a dict with _target_ key
    trainer.config = OmegaConf.to_container(cfg, resolve=True),

    logger.info(f"Starting training...")
    trainer.train(cfg.trainer.epochs)

    logger.info("Training completed!")

    return 0



if __name__ == "__main__":

    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env")


    def githash(*args, **kwargs) -> str:
        try:
            import subprocess
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            return git_hash
        except Exception as e:
            logger.warning(f"Could not retrieve git hash: {e}")
            return "unknown"


    OmegaConf.register_new_resolver("mod", lambda x, y: x % y)
    OmegaConf.register_new_resolver("div", lambda x, y: int(x / y))
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("githash", githash)
    main()