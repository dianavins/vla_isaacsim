from edgevla.devices import DeviceProfile, ORIN_NANO_8GB, ORIN_NX_16GB


def test_effective_ops_applies_utilization():
    dev = DeviceProfile(
        name="synthetic",
        mem_budget_bytes=8_000_000_000,
        int8_tops=20.0,
        utilization=0.5,
        power_w=15.0,
    )
    # 20 TOPS * 0.5 = 10e12 ops/s
    assert dev.effective_ops_per_s() == 10e12


def test_estimated_profiles_present_and_ordered():
    assert ORIN_NANO_8GB.mem_budget_bytes < ORIN_NX_16GB.mem_budget_bytes
    assert ORIN_NANO_8GB.name == "orin_nano_8gb"
    assert ORIN_NX_16GB.name == "orin_nx_16gb"
