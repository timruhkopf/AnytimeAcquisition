import torch
from tqdm import tqdm

from src.callbacks.plot_trajectories_callback import PlotTrajectoriesCallback


class DefaultTrainer:
    def __init__(self, model, optimizer, loss_fn, env, callbacks=[PlotTrajectoriesCallback()],
                 device="cpu", logger=None,
                 **kwargs):
        self.model = model.to(device)
        self.optimizer = optimizer(self.model.parameters())
        self.loss_fn = loss_fn
        self.env = env

        self.logger = logger
        self.callbacks = callbacks or []
        self.device = device
        self.epoch = 0
        self.batch_idx = 0
        self.last_loss = None
        self.train_loader = None
        self.kwargs = kwargs
        for cb in self.callbacks:
            cb.set_trainer(self)

    def train(self, num_epochs, B=32):

        B = self.kwargs.get('B', B)
        T = self.kwargs.get('T', 25)

        for cb in self.callbacks: cb.on_train_begin()
        for epoch in tqdm(range(num_epochs), desc="Training epochs"):
            self.epoch = epoch
            for cb in self.callbacks: cb.on_epoch_begin()

            self.model.eval()


            X = self.model.explore(
                step=epoch,
                # FIXME: repeats should be a parameter
                env=self.env, B=B, T=T,
            )

            self.train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X),
                batch_size=B,
                shuffle=True
            )

            self.model.train()
            # fixed environment evaluation allows to reiterate over the same trajectory
            for batch_idx, X in enumerate(self.train_loader):
                self.batch_idx = batch_idx
                X = X[0].to(self.device)
                pred, _ = self.model(X)

                y = self.env.evaluate(pred[..., :-1])
                loss = self.loss_fn(pred, y, alternatives=X)
                self.last_loss = loss.item()

                if self.logger is not None:
                    self.logger.log({'epoch': epoch, 'batch': batch_idx, 'loss': self.last_loss})

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                for cb in self.callbacks: cb.on_batch_end()

            # reset env
            # self.env.resample()

            for cb in self.callbacks: cb.on_epoch_end()
        for cb in self.callbacks: cb.on_train_end()
