from __future__ import annotations

from edgevla.jetson import GateVerdict


def success_exit(
    verdict: GateVerdict,
    success: float,
    baseline_success: float,
    retain: float = 0.90,
) -> bool:
    return verdict.all_pass() and success >= retain * baseline_success


def diminishing_returns_exit(
    hv_history: list[float],
    rel_threshold: float = 0.02,
    window: int = 2,
) -> bool:
    if len(hv_history) <= window:
        return False
    past = hv_history[-1 - window]
    current = hv_history[-1]
    if past <= 0:
        return False
    rel_gain = (current - past) / past
    return rel_gain < rel_threshold
