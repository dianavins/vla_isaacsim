from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KPIRow:
    model_name: str
    device_name: str
    success_avg: float
    success_per_suite: dict[str, float]
    control_rate_hz: float
    latency_ms_p50: float
    latency_ms_p95: float
    peak_mem_gb: float
    params_m: float
    disk_size_mb: float
    energy_j: float | None = None
    avg_power_w: float | None = None
    estimated: frozenset[str] = field(default_factory=frozenset)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "device_name": self.device_name,
            "success_avg": self.success_avg,
            "success_per_suite": dict(self.success_per_suite),
            "control_rate_hz": self.control_rate_hz,
            "latency_ms_p50": self.latency_ms_p50,
            "latency_ms_p95": self.latency_ms_p95,
            "peak_mem_gb": self.peak_mem_gb,
            "params_m": self.params_m,
            "disk_size_mb": self.disk_size_mb,
            "energy_j": self.energy_j,
            "avg_power_w": self.avg_power_w,
            "estimated": sorted(self.estimated),
        }
