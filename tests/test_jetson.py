from edgevla.devices import DeviceProfile
from edgevla.jetson import three_light_gate
from edgevla.modelstats import ModelStats


# Synthetic device: 8 GB budget, 10e12 effective ops/s, 15 W.
DEV = DeviceProfile(
    name="synthetic",
    mem_budget_bytes=8_000_000_000,
    int8_tops=20.0,
    utilization=0.5,
    power_w=15.0,
)


def test_small_fast_model_passes_all_three_lights():
    # peak mem 2 GB; 1e9 flops/step -> 10000 Hz; 15 W <= 15 W
    stats = ModelStats(
        params=10_000_000,
        weight_bytes=1_000_000_000,
        activation_bytes=900_000_000,
        state_bytes=100_000_000,
        flops_per_step=1_000_000_000,
    )
    v = three_light_gate(stats, DEV)
    assert v.lights() == (True, True, True)
    assert v.all_pass() is True
    assert v.peak_mem_gb == 2.0
    assert v.control_rate_hz == 10000.0


def test_fits_but_too_slow_is_the_litevla_edge_failure_mode():
    # peak mem 2 GB (fits); 2e12 flops/step -> 5 Hz (< 10 Hz)
    stats = ModelStats(
        params=7_000_000_000,
        weight_bytes=1_500_000_000,
        activation_bytes=400_000_000,
        state_bytes=100_000_000,
        flops_per_step=2_000_000_000_000,
    )
    v = three_light_gate(stats, DEV)
    assert v.fits is True
    assert v.realtime is False
    assert v.control_rate_hz == 5.0
    assert v.all_pass() is False


def test_over_memory_budget_fails_fits():
    stats = ModelStats(
        params=7_000_000_000,
        weight_bytes=9_000_000_000,
        activation_bytes=0,
        state_bytes=0,
        flops_per_step=1_000_000_000,
    )
    v = three_light_gate(stats, DEV)
    assert v.fits is False


def test_power_budget_can_fail_independently():
    stats = ModelStats(
        params=10_000_000,
        weight_bytes=1_000_000_000,
        activation_bytes=0,
        state_bytes=0,
        flops_per_step=1_000_000_000,
    )
    v = three_light_gate(stats, DEV, power_budget_w=10.0)  # device draws 15 W
    assert v.fits is True
    assert v.realtime is True
    assert v.power_ok is False
