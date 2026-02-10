from typing import Dict


class AbstractCallback:
    """Base class for training callbacks."""
    def __init__(self, verbose:bool= False):
        self.verbose = verbose
        self.trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_trainer_init(self, **kwargs):
        """Called at the end of trainer initialization; allows e.g. the checkpoint callback to load the trainer checkpoint."""
        pass

    def on_train_start(self, **kwargs):
        pass

    def on_epoch_start(self, epoch: int, **kwargs):
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        """Called at the end of an epoch. Can return a dict of metrics to log.
        These will be new entries to the trainer's metrics dict after all on_epoch_end
        calls are done. The final and complete dict will be passed to log_on_epoch_end."""
        pass

    def log_on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        """This allows us to separate logging from other epoch end actions; this way
        we know for sure, that the on_epoch_end computations are done before logging."""
        pass

    def on_forward_end(self, batch, single_eval_pos, output, targets) -> Dict:
        pass

    def on_step_end(self, epoch: int, step: int, metrics: Dict[str, float], **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def log_on_train_end(self, **kwargs):
        pass

    def on_clipping(self, epoch: int, step: int, metrics: Dict[str, float], **kwargs) -> Dict:
        pass


class CallbackHandler:
    """On event occurence, call all registered callbacks."""

    def __init__(self, callbacks, trainer):
        self.callbacks = callbacks
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_event(self, event_name: str, *args, **kwargs):
        feedback = {}
        for callback in self.callbacks:
            method = getattr(callback, event_name, None)
            D = method(*args, **kwargs)
            if D is not None:
                feedback.update(D)

        return feedback