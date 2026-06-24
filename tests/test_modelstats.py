from edgevla.modelstats import ModelStats


def test_peak_mem_sums_components_plus_overhead():
    s = ModelStats(
        params=450_000_000,
        weight_bytes=900_000_000,
        activation_bytes=300_000_000,
        state_bytes=50_000_000,
        flops_per_step=2_000_000_000,
    )
    assert s.peak_mem_bytes() == 1_250_000_000
    assert s.peak_mem_bytes(overhead_bytes=250_000_000) == 1_500_000_000
