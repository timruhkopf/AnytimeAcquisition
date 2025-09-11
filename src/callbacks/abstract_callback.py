import abc


class AbstractCallback(abc.ABC):

    # if self does not have attribute, search in trainer
    def __getattr__(self, name):
        if 'trainer' in self.__dict__ and hasattr(self.trainer, name):
            return getattr(self.trainer, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, trainer): pass
    def on_train_end(self, trainer): pass
    def on_epoch_begin(self, trainer): pass
    def on_epoch_end(self, trainer): pass
    def on_batch_end(self, trainer): pass
