import torch
from tqdm import tqdm


class DefaultTrainer:
    def __init__(self, model, optimizer, loss_fn, env, callbacks=None, device="cpu", logger=None,
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

    def train(self, num_epochs, B=16):

        B = self.kwargs.get('B', B)
        T = self.kwargs.get('T', 24)

        for cb in self.callbacks: cb.on_train_begin()
        for epoch in tqdm(range(num_epochs), desc="Training epochs"):
            self.epoch = epoch
            for cb in self.callbacks: cb.on_epoch_begin()

            self.model.eval()

            initial_condition = self.env.sample_initial_condition(B=B)

            X = self.model.generate(
                self.env, B=B, T=T,
                initial_condition=initial_condition
            )

            if hasattr(self.model, 'explore'):
                exploration_X = self.model.explore(
                    self.env, B=B, T=T,
                    initial_condition=initial_condition
                )
                X = torch.cat([X, exploration_X], dim=0)

            b = X.shape[0]

            self.train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X),
                batch_size=self.kwargs.get('batch_size', b ),
                shuffle=True
            )

            self.model.train()
            for batch_idx, X in enumerate(self.train_loader):
                self.batch_idx = batch_idx
                X = X[0].to(self.device)
                pred, _ = self.model(X[:, :-1, :])

                y = self.env.evaluate(pred)

                loss = self.loss_fn(y)
                self.last_loss = loss.item()
                if self.logger is not None:
                    self.logger.log({'epoch': epoch, 'batch': batch_idx, 'loss': self.last_loss})

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                for cb in self.callbacks: cb.on_batch_end()
            for cb in self.callbacks: cb.on_epoch_end()
        for cb in self.callbacks: cb.on_train_end()
