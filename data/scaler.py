import numpy as np


class StandardScalerNP:
    """Z-score нормализация для numpy массивов."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScalerNP":
        self.mean_ = X.mean(axis=0)
        self.std_ = np.maximum(X.std(axis=0), self.eps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler не обучен. Вызовите fit() сначала.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def state_dict(self) -> dict:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler не обучен.")
        return {"mean": self.mean_, "std": self.std_, "eps": self.eps}

    @classmethod
    def from_state_dict(cls, state: dict) -> "StandardScalerNP":
        obj = cls(eps=state.get("eps", 1e-6))
        obj.mean_ = state["mean"]
        obj.std_ = state["std"]
        return obj
