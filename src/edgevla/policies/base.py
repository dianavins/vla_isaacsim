from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VLAPolicy(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, observation: object) -> np.ndarray: ...
