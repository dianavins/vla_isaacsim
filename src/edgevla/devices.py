from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeviceProfile:
    name: str
    mem_budget_bytes: int
    int8_tops: float
    utilization: float
    power_w: float

    def effective_ops_per_s(self) -> float:
        return self.int8_tops * 1e12 * self.utilization


# Nominal [EST] constants — conservative, tunable. Not measured (no board).
# Usable unified memory is below nameplate (OS + framework reserve).
ORIN_NANO_8GB = DeviceProfile(
    name="orin_nano_8gb",
    mem_budget_bytes=6_500_000_000,   # ~6.5 GB usable of 8 GB
    int8_tops=20.0,                   # conservative dense-INT8 effective
    utilization=0.3,                  # derate for real workloads
    power_w=15.0,
)

ORIN_NX_16GB = DeviceProfile(
    name="orin_nx_16gb",
    mem_budget_bytes=14_000_000_000,  # ~14 GB usable of 16 GB
    int8_tops=70.0,
    utilization=0.3,
    power_w=20.0,
)
