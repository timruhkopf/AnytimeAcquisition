"""Placeholder trainer used to prove the Hydra wiring end to end."""
from anytimeacquisition.callbacks.handler import Callback, CallbackHandler


class DummyTrainer:
    def __init__(self, benchmark, prior, surrogate, n_steps: int = 3, callbacks: list[Callback] | None = None):
        self.benchmark = benchmark
        self.prior = prior
        self.surrogate = surrogate
        self.n_steps = n_steps
        # Same injectable-metrics mechanism as trainer/pfn_trainer.py's
        # PFNTrainer (see callbacks/handler.py) -- this loop is a one-shot
        # placeholder (no per-step cadence yet), so callbacks just run once
        # against the fitted surrogate before returning.
        self.callback_handler = CallbackHandler(callbacks)

    def run(self) -> dict:
        xs = self.prior.sample(self.n_steps)
        ys = [self.benchmark.evaluate([x]) for x in xs]
        self.surrogate.fit(xs, ys)
        result = {"n_steps": self.n_steps, "best_y": min(ys)}
        result.update(self.callback_handler.run(step=0, trainer=self, default_every_n_steps=1))
        return result
