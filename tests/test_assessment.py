import numpy as np

from edgevla.assessment import kpi_assessment
from edgevla.devices import DeviceProfile
from edgevla.envs.fake import FakeEnv
from edgevla.modelstats import ModelStats
from edgevla.policies.dummy import DummyPolicy


DEV = DeviceProfile(
    name="synthetic",
    mem_budget_bytes=8_000_000_000,
    int8_tops=20.0,
    utilization=0.5,
    power_w=15.0,
)
STATS = ModelStats(
    params=450_000_000,
    weight_bytes=900_000_000,
    activation_bytes=900_000_000,
    state_bytes=200_000_000,
    flops_per_step=1_000_000_000,
)


def make_fake_timer(step_seconds=0.01):
    """Deterministic clock: advances `step_seconds` on every call."""
    t = {"now": 0.0}

    def timer():
        t["now"] += step_seconds
        return t["now"]

    return timer


def test_success_rate_matches_scripted_pattern():
    # 2 episodes/suite, pattern [True, False] -> 50% per suite.
    env = FakeEnv(suites=["spatial"], steps_per_episode=1, success_pattern=[True, False])
    policy = DummyPolicy(action_dim=7)
    row, verdict = kpi_assessment(
        policy, env, STATS, DEV,
        episodes_per_suite=2, max_steps=5,
        peak_mem_gb=2.0, disk_size_mb=900.0,
    )
    assert row.success_per_suite["spatial"] == 0.5
    assert row.success_avg == 0.5


def test_latency_and_control_rate_from_timer():
    env = FakeEnv(suites=["spatial"], steps_per_episode=1, success_pattern=[True])
    policy = DummyPolicy(action_dim=7)
    # Each act() brackets two timer() calls: t0=0.01, t1=0.02 -> 0.01 s = 10 ms.
    row, _ = kpi_assessment(
        policy, env, STATS, DEV,
        episodes_per_suite=1, max_steps=1,
        peak_mem_gb=2.0, disk_size_mb=900.0,
        timer=make_fake_timer(0.01),
    )
    assert round(row.latency_ms_p50, 3) == 10.0
    assert round(row.control_rate_hz, 3) == 100.0  # 1000 / 10 ms


def test_verdict_is_jetson_estimate_and_desktop_row_not_estimated():
    env = FakeEnv(suites=["spatial"], steps_per_episode=1, success_pattern=[True])
    policy = DummyPolicy(action_dim=7)
    row, verdict = kpi_assessment(
        policy, env, STATS, DEV,
        episodes_per_suite=1, max_steps=1,
        peak_mem_gb=2.0, disk_size_mb=900.0,
    )
    assert row.estimated == frozenset()            # desktop numbers measured
    assert verdict.control_rate_hz == 10000.0      # 10e12 ops / 1e9 flops
    assert row.params_m == 450.0
