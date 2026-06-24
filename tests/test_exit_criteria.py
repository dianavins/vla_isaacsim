from edgevla.compression.exit_criteria import (
    diminishing_returns_exit,
    success_exit,
)
from edgevla.jetson import GateVerdict


def passing_verdict():
    return GateVerdict(
        fits=True, realtime=True, power_ok=True,
        peak_mem_gb=2.0, control_rate_hz=30.0, avg_power_w=10.0, energy_j=0.3,
    )


def failing_verdict():
    return GateVerdict(
        fits=True, realtime=False, power_ok=True,
        peak_mem_gb=2.0, control_rate_hz=6.6, avg_power_w=10.0, energy_j=1.5,
    )


def test_success_exit_needs_all_lights_and_retained_success():
    base = 0.80
    assert success_exit(passing_verdict(), success=0.73, baseline_success=base) is True
    assert success_exit(passing_verdict(), success=0.70, baseline_success=base) is False
    assert success_exit(failing_verdict(), success=0.80, baseline_success=base) is False


def test_diminishing_returns_fires_on_flat_frontier():
    # Big gains early, then plateau within 2%.
    improving = [100.0, 150.0, 220.0]
    assert diminishing_returns_exit(improving) is False
    plateau = [220.0, 222.0, 223.0]  # 223/220 - 1 = 1.36% < 2%
    assert diminishing_returns_exit(plateau) is True


def test_diminishing_returns_needs_enough_history():
    assert diminishing_returns_exit([100.0, 101.0]) is False  # only 2 entries, window=2
