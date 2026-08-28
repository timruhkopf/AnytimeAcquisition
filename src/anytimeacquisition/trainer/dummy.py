"""Placeholder trainer used to prove the Hydra wiring end to end."""


class DummyTrainer:
    def __init__(self, benchmark, prior, surrogate, n_steps: int = 3):
        self.benchmark = benchmark
        self.prior = prior
        self.surrogate = surrogate
        self.n_steps = n_steps

    def run(self) -> dict:
        xs = self.prior.sample(self.n_steps)
        ys = [self.benchmark.evaluate([x]) for x in xs]
        self.surrogate.fit(xs, ys)
        return {"n_steps": self.n_steps, "best_y": min(ys)}
