from __future__ import annotations

from dataclasses import dataclass

from edgevla.devices import DeviceProfile
from edgevla.modelstats import ModelStats


@dataclass
class GateVerdict:
    fits: bool
    realtime: bool
    power_ok: bool
    peak_mem_gb: float
    control_rate_hz: float
    avg_power_w: float
    energy_j: float

    def lights(self) -> tuple[bool, bool, bool]:
        return (self.fits, self.realtime, self.power_ok)

    def all_pass(self) -> bool:
        return self.fits and self.realtime and self.power_ok


def three_light_gate(
    stats: ModelStats,
    device: DeviceProfile,
    *,
    overhead_bytes: int = 0,
    rt_floor_hz: float = 10.0,
    power_budget_w: float = 15.0,
) -> GateVerdict:
    peak_mem = stats.peak_mem_bytes(overhead_bytes)
    fits = peak_mem <= device.mem_budget_bytes

    control_rate = device.effective_ops_per_s() / stats.flops_per_step
    realtime = control_rate >= rt_floor_hz

    avg_power = device.power_w
    latency_s = 1.0 / control_rate
    energy = avg_power * latency_s
    power_ok = avg_power <= power_budget_w

    return GateVerdict(
        fits=fits,
        realtime=realtime,
        power_ok=power_ok,
        peak_mem_gb=peak_mem / 1e9,
        control_rate_hz=control_rate,
        avg_power_w=avg_power,
        energy_j=energy,
    )
