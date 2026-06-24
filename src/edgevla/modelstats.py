from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelStats:
    params: int
    weight_bytes: int
    activation_bytes: int
    state_bytes: int
    flops_per_step: int

    def peak_mem_bytes(self, overhead_bytes: int = 0) -> int:
        return (
            self.weight_bytes
            + self.activation_bytes
            + self.state_bytes
            + overhead_bytes
        )
