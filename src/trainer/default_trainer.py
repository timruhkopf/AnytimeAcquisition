from functools import partial

import torch
from tqdm import tqdm

import inspect


def has_key_in_init(cls, key):
    sig = inspect.signature(cls.__init__)
    # Exclude 'self' and check if 'env' is present
    params = [p for p in sig.parameters.values() if p.name != 'self']
    return key in [p.name for p in params]


class DefaultTrainer:
    def __init__(self, model, optimizer, loss_fn, env,
                 callbacks=None,
                 device="cpu", logger=None,
                 **kwargs):
        self.model = model.to(device)
        self.optimizer = optimizer(self.model.parameters())
        self.loss_fn = loss_fn
        if isinstance(loss_fn, partial) and has_key_in_init(loss_fn.func, 'env'):
            self.loss_fn = self.loss_fn(env=env)
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
                # FIXME: n_alternatives should be a parameter
                env=self.env, B=B, T=T,
            )

            self.train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X),
                batch_size=B,
                shuffle=True
            )

            self.model.train()

            # fixed environment evaluation allows to reiterate over the same trajectory
            # otherwise, we will just have a single (batched) pass over an env instance
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

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.optimizer.zero_grad()

                for cb in self.callbacks: cb.on_batch_end()

            # reset env
            self.env.resample()

            for cb in self.callbacks: cb.on_epoch_end()
        for cb in self.callbacks: cb.on_train_end()
