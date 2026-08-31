"""Generic training-callback mechanism, shared across `trainer/*.py`
classes -- lets a caller inject arbitrary periodic metric-computation hooks
(e.g. validation performance on a real benchmark, a special edge-case
check) without the trainer's own loop needing to know about them ahead of
time. `PFNTrainer`/`DummyTrainer` each own a `CallbackHandler` and call
`self.callbacks.run(step, self, self.log_every)` wherever they already log
their own built-in metrics -- see those modules.

Not Hydra-`_target_`-instantiable on its own (`configs/callbacks/` is
MLflow tracking settings, unrelated to this despite the shared name): a
`Callback` is typically a closure built in the caller's own script/
notebook/pipeline `main()` -- it needs to close over whatever it validates
against (a specific real-benchmark instance, a specific edge case), so it
isn't a named, swappable component the way priors/models/trainers are.
"""
from typing import Any, Callable


class Callback:
    """One periodic hook. `fn(step, trainer)` -> a flat {metric_name: value}
    dict (empty or None to skip logging this call, e.g. a check that only
    asserts/prints). `trainer` is the owning trainer instance itself, so
    `fn` can reach whatever state it needs (`.model`, `.prior`, `.bar_dist`,
    ...) without every piece of state a future callback might want having
    to be threaded through the callback signature ahead of time.

    Metrics are namespaced under `name` (e.g. name="real_benchmark" ->
    "real_benchmark/regret") -- same MLflow dashboard-grouping convention as
    `trainer/pfn_trainer.py`'s own `nll/train`, `mse/train`. `name=""`
    (falsy) skips this prefixing entirely -- for a callback whose `fn`
    already returns fully-namespaced keys itself (e.g. one hook computing
    several metric *types* across several probes/sources at once, where
    grouping by metric type first -- `nll/val_dim1`, `nll/val_dim2`, ... --
    reads better on a dashboard than grouping by source first; see
    `callbacks/dim_validation.py`).

    `every_n_steps` defaults to the trainer's own logging cadence (None);
    set it explicitly for a callback that should run coarser (an expensive
    real-benchmark eval) or finer than the main loop's own metrics.

    Deliberately NOT a `@dataclass`: a list of these gets passed straight
    through `hydra.utils.instantiate(cfg.trainer, ..., callbacks=callbacks)`
    (see `pipelines/train_pfn.py`'s `main()`) as a plain extra kwarg, and
    Hydra/OmegaConf auto-converts `dataclass`/attrs instances it finds
    there into structured configs -- which strips real methods like
    `maybe_run` below, breaking at the first call with a confusing
    `ConfigAttributeError: Key 'maybe_run' not in 'Callback'`. A plain class
    isn't OmegaConf-representable, so it passes through untouched.
    """

    def __init__(self, name: str, fn: Callable[[int, Any], dict | None], every_n_steps: int | None = None):
        self.name = name
        self.fn = fn
        self.every_n_steps = every_n_steps

    def maybe_run(self, step: int, trainer: Any, default_every_n_steps: int) -> dict:
        every = self.every_n_steps or default_every_n_steps
        if every <= 0 or step % every != 0:
            return {}
        result = self.fn(step, trainer)
        if not result:
            return {}
        if not self.name:
            return dict(result)
        return {f"{self.name}/{k}": v for k, v in result.items()}


class CallbackHandler:
    """Owns a list of `Callback`s and runs the ones due at a given step,
    merging their namespaced metrics into one dict."""

    def __init__(self, callbacks: list[Callback] | None = None):
        self.callbacks = list(callbacks or [])

    def run(self, step: int, trainer: Any, default_every_n_steps: int) -> dict:
        metrics: dict = {}
        for callback in self.callbacks:
            metrics.update(callback.maybe_run(step, trainer, default_every_n_steps))
        return metrics


if __name__ == "__main__":
    # Two callbacks at different cadences against a fake "trainer" (just
    # needs to be something callbacks can read state off of) -- shows
    # namespacing and independent cadence, not tied to any real trainer.
    class _FakeTrainer:
        def __init__(self):
            self.loss = 1.0

    trainer = _FakeTrainer()

    def cheap_check(step: int, t: "_FakeTrainer") -> dict:
        return {"loss_seen": t.loss}

    def expensive_real_benchmark_eval(step: int, t: "_FakeTrainer") -> dict:
        return {"regret": 1.0 / (step + 1)}

    handler = CallbackHandler([
        Callback(name="cheap", fn=cheap_check),  # every_n_steps=None -> uses default_every_n_steps
        Callback(name="real_benchmark", fn=expensive_real_benchmark_eval, every_n_steps=4),
    ])

    for step in range(8):
        trainer.loss = 1.0 - step * 0.1
        metrics = handler.run(step, trainer, default_every_n_steps=2)
        print(f"step {step}: {metrics}")

    print("\n'cheap/loss_seen' fires every 2 steps, 'real_benchmark/regret' every 4 -- "
          "check the printed steps above match that.")
