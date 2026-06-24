from __future__ import annotations

import numpy as np


class DummyPolicy:
    def __init__(self, action_dim: int, name: str = "dummy") -> None:
        self.action_dim = action_dim
        self.name = name

    def reset(self) -> None:
        return None

    def act(self, observation: object) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float32)
