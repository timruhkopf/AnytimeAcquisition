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
        # Pre-build the cache: { 'on_batch_end': [method1, method2], ... }
        self._method_cache = self._build_cache()

    def _is_functional(self, method):
        """Checks if a method actually contains logic (more than just 'pass')."""
        if not method or not callable(method):
            return False

        code = getattr(method, "__code__", None)
        if not code:
            return True  # Keep C-extensions or built-ins

        # Bytecode > 4 bytes filters out 'pass' and 'return None'
        return len(code.co_code) > 4

    def _build_cache(self):
        """Scans all callbacks once to map implemented events."""

        # find all methods from AbstractCallback
        all_events = {func for func in dir(AbstractCallback) if
                      callable(getattr(AbstractCallback, func)) and not func.startswith("_")}
        all_events.remove("set_trainer")

        cache = {}
        for cb in self.callbacks:
            # 1. Initialize the trainer on the callback
            cb.set_trainer(self.trainer)


            # 2. Find all public methods (excluding dunder methods)
            for attr_name in set(dir(cb)).intersection(all_events):

                method = getattr(cb, attr_name)
                if self._is_functional(method):
                    if attr_name not in cache:
                        cache[attr_name] = []
                    cache[attr_name].append(method)
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