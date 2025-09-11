import wandb

from src.callbacks.abstract_callback import AbstractCallback


class WandbLogger(AbstractCallback):
    def __init__(self, project="transformer-regression"):
        self.run = wandb.init(project=project)

    def on_batch_end(self, trainer):
        wandb.log({
            "train/loss": trainer.last_loss,
            "epoch": trainer.epoch,
            "batch": trainer.batch_idx,
        })

    def on_train_end(self, trainer):
        wandb.finish()
