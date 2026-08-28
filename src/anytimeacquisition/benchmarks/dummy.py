"""Placeholder benchmark used to prove the Hydra wiring end to end."""


class DummyBenchmark:
    def __init__(self, dim: int = 2):
        self.dim = dim

    def evaluate(self, x: list[float]) -> float:
        return sum(xi**2 for xi in x)
