"""Placeholder surrogate used to prove the Hydra wiring end to end."""


class DummySurrogate:
    def __init__(self):
        self.fitted = False

    def fit(self, x: list[float], y: list[float]) -> None:
        self.fitted = True

    def predict(self, x: list[float]) -> list[float]:
        return [0.0 for _ in x]
