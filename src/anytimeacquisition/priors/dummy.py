"""Placeholder prior/environment used to prove the Hydra wiring end to end."""
import random


class DummyPrior:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def sample(self, n: int) -> list[float]:
        rng = random.Random(self.seed)
        return [rng.random() for _ in range(n)]
