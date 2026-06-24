from __future__ import annotations

import statistics
import time

from edgevla.devices import DeviceProfile
from edgevla.jetson import GateVerdict, three_light_gate
from edgevla.kpi import KPIRow
from edgevla.modelstats import ModelStats


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def kpi_assessment(
    policy,
    env,
    stats: ModelStats,
    device: DeviceProfile,
    *,
    episodes_per_suite: int,
    max_steps: int,
    overhead_bytes: int = 0,
    rt_floor_hz: float = 10.0,
    power_budget_w: float = 15.0,
    peak_mem_gb: float | None = None,
    disk_size_mb: float | None = None,
    timer=time.perf_counter,
) -> tuple[KPIRow, GateVerdict]:
    success_per_suite: dict[str, float] = {}
    latencies_ms: list[float] = []

    for suite in env.suites:
        successes = 0
        for _ in range(episodes_per_suite):
            env.reset(suite)
            policy.reset()
            for _ in range(max_steps):
                obs = {"suite": suite}
                t0 = timer()
                policy.act(obs)
                t1 = timer()
                latencies_ms.append((t1 - t0) * 1000.0)
                _, _, done, _ = env.step(_zero_like(policy))
                if done:
                    break
            if env.is_success():
                successes += 1
        success_per_suite[suite] = successes / episodes_per_suite

    success_avg = statistics.fmean(success_per_suite.values())
    p50 = _percentile(latencies_ms, 0.50)
    p95 = _percentile(latencies_ms, 0.95)
    control_rate = 1000.0 / p50 if p50 > 0 else 0.0

    if peak_mem_gb is None:
        peak_mem_gb = stats.peak_mem_bytes(overhead_bytes) / 1e9
    if disk_size_mb is None:
        disk_size_mb = stats.weight_bytes / 1e6

    row = KPIRow(
        model_name=getattr(policy, "name", "unknown"),
        device_name=device.name,
        success_avg=success_avg,
        success_per_suite=success_per_suite,
        control_rate_hz=control_rate,
        latency_ms_p50=p50,
        latency_ms_p95=p95,
        peak_mem_gb=peak_mem_gb,
        params_m=stats.params / 1e6,
        disk_size_mb=disk_size_mb,
        estimated=frozenset(),  # desktop-measured row
    )

    verdict = three_light_gate(
        stats,
        device,
        overhead_bytes=overhead_bytes,
        rt_floor_hz=rt_floor_hz,
        power_budget_w=power_budget_w,
    )
    return row, verdict


def _zero_like(policy):
    import numpy as np

    dim = getattr(policy, "action_dim", 1)
    return np.zeros(dim, dtype=np.float32)
