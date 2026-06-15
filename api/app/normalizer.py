from __future__ import annotations

from pathlib import Path
import numpy as np


class MinMaxNormalizer:
    def __init__(self, min_a: np.ndarray, max_a: np.ndarray, eps: float = 1e-4):
        self.min_a = np.asarray(min_a, dtype=np.float64)
        self.max_a = np.asarray(max_a, dtype=np.float64)
        self.eps = eps

        if self.min_a.shape != self.max_a.shape:
            raise ValueError(
                f"min_a shape {self.min_a.shape} != max_a shape {self.max_a.shape}"
            )

    @classmethod
    def from_file(cls, path: str | Path, eps: float = 1e-4):
        obj = np.load(path, allow_pickle=True)
        min_a = obj["min_a"]
        max_a = obj["max_a"]
        return cls(min_a=min_a, max_a=max_a, eps=eps)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (x - self.min_a) / (self.max_a - self.min_a + self.eps)