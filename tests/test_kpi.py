from edgevla.kpi import KPIRow


def make_row(**over):
    base = dict(
        model_name="smolvla",
        device_name="rtx4090",
        success_avg=0.76,
        success_per_suite={"spatial": 0.84, "object": 0.88},
        control_rate_hz=12.0,
        latency_ms_p50=80.0,
        latency_ms_p95=110.0,
        peak_mem_gb=2.1,
        params_m=450.0,
        disk_size_mb=900.0,
    )
    base.update(over)
    return KPIRow(**base)


def test_defaults_have_no_estimated_fields():
    row = make_row()
    assert row.energy_j is None
    assert row.avg_power_w is None
    assert row.estimated == frozenset()


def test_to_dict_roundtrips_core_fields():
    row = make_row(energy_j=0.5, avg_power_w=15.0, estimated=frozenset({"energy_j"}))
    d = row.to_dict()
    assert d["model_name"] == "smolvla"
    assert d["success_avg"] == 0.76
    assert d["estimated"] == ["energy_j"]
    assert d["success_per_suite"]["spatial"] == 0.84
