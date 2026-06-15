from collections import deque
import numpy as np


class ProcessedWindowBuffer:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def push(self, vec: np.ndarray) -> None:
        self.buffer.append(vec)

    def ready(self) -> bool:
        return len(self.buffer) == self.window_size

    def get(self) -> np.ndarray:
        return np.stack(self.buffer, axis=0)