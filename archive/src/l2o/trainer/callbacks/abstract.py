from typing import Dict
import inspect


class AbstractCallback:
    """Base class for training callbacks."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    @property
    def epoch(self):
        return self.trainer.epoch


    def on_trainer_init(self, **kwargs):
        """Called at the end of trainer initialization; allows e.g. the checkpoint callback to load the trainer checkpoint."""
        pass

    def on_train_start(self, **kwargs):
        pass

    def on_epoch_start(self, **kwargs):
        pass

    def on_policy_epoch_end(self,  metrics: Dict[str, float], **kwargs):
        pass

    def on_policy_epoch_start(self,  **kwargs):
        pass

    def on_rollout_end(self, **kwargs):
        pass

    def on_policy_clipping(self, **kwargs) -> Dict:
        pass

    def on_epoch_end(self, metrics: Dict[str, float], **kwargs):
        """Called at the end of an epoch. Can return a dict of metrics to log.
        These will be new entries to the trainer's metrics dict after all on_epoch_end
        calls are done. The final and complete dict will be passed to log_on_epoch_end."""
        pass

    def log_on_epoch_end(self, metrics: Dict[str, float], **kwargs):
        """This allows us to separate logging from other epoch end actions; this way
        we know for sure, that the on_epoch_end computations are done before logging."""
        pass

    def on_forward_end(self, output, targets) -> Dict:
        pass

    def on_step_end(self, step: int, metrics: Dict[str, float], **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def log_on_train_end(self, **kwargs):
        pass

    def on_clipping(self, step: int, metrics: Dict[str, float], **kwargs) -> Dict:
        pass


class CallbackHandler:
    """On event occurrence, call only callbacks with actual implementations."""

    def __init__(self, callbacks, trainer):
        self.callbacks = callbacks
        self.trainer = trainer

        for cb in self.callbacks:
            # Initialize the trainer on the callback
            cb.set_trainer(self.trainer)

        # Pre-build the cache: { 'on_batch_end': [method1, method2], ... }
        self._method_cache = self._build_cache()



    def _build_cache(self):
        """Scans all callbacks once to map implemented events by comparing method objects."""

        # 1. Identify all valid event methods from the base class
        # We filter for public methods that don't start with '_' or 'set_trainer'
        all_events = [
            method_name for method_name in dir(AbstractCallback)
            if callable(getattr(AbstractCallback, method_name))
               and not method_name.startswith("_")
               and method_name != "set_trainer"
        ]

        cache = {}
        for cb in self.callbacks:

            for event_name in all_events:
                # Get the method from the current callback instance
                cb_method = getattr(cb, event_name)
                # Get the base method from the Abstract class
                base_method = getattr(AbstractCallback, event_name)

                # SAFER CHECK:
                # If the function objects are different, it means the subclass
                # provided its own implementation.
                if cb_method.__func__ is not base_method:
                    if event_name not in cache:
                        cache[event_name] = []
                    cache[event_name].append(cb_method)

        return cache

    def on_event(self, event_name: str, *args, **kwargs):
        feedback = {}

        # O(1) lookup: No inspection, no getattr, just execution
        methods = self._method_cache.get(event_name, [])

        for method in methods:
            res = method(*args, **kwargs)
            if isinstance(res, dict):
                feedback.update(res)

        return feedback