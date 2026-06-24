from edgevla.compression.loop import Variant, compression_loop
from edgevla.compression.pareto import ParetoPoint
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


def test_loop_exits_on_success_when_a_variant_passes_gates_and_keeps_success():
    def design(round_idx, evidence):
        return ["strategyA"]

    def sweep(candidates):
        # A small, fast model that retains 95% of an 0.80 baseline (0.76 >= 0.72).
        return [Variant(ParetoPoint(250, 0.76, "win"), passing_verdict(), 0.76)]

    result = compression_loop(
        design=design, sweep=sweep,
        baseline_success=0.80, ref_size_mb=1000.0,
    )
    assert result.exit_reason == "success"
    assert result.best.label == "win"
    assert result.rounds == 1


def test_loop_exits_on_diminishing_returns_when_frontier_plateaus():
    # Each round adds a variant that fails real-time (never a success exit),
    # and the Pareto gains shrink below 2% so the loop gives up.
    schedule = [
        ParetoPoint(500, 0.60, "r0"),  # hv0
        ParetoPoint(400, 0.61, "r1"),  # modest gain
        ParetoPoint(399, 0.611, "r2"),  # tiny gain -> plateau
        ParetoPoint(398, 0.612, "r3"),
    ]
    calls = {"i": 0}

    def design(round_idx, evidence):
        return [round_idx]

    def sweep(candidates):
        p = schedule[calls["i"]]
        calls["i"] += 1
        return [Variant(p, failing_verdict(), p.success)]

    result = compression_loop(
        design=design, sweep=sweep,
        baseline_success=0.80, ref_size_mb=1000.0,
        max_rounds=10,
    )
    assert result.exit_reason == "diminishing_returns"
    assert result.best is not None  # best frontier point recorded on give-up


def test_loop_passes_previous_variants_as_evidence():
    seen = []

    def design(round_idx, evidence):
        seen.append(evidence)
        return [round_idx]

    def sweep(candidates):
        return [Variant(ParetoPoint(500, 0.50, "x"), failing_verdict(), 0.50)]

    compression_loop(
        design=design, sweep=sweep,
        baseline_success=0.80, ref_size_mb=1000.0,
        max_rounds=2,
    )
    assert seen[0] is None                       # round 0: no evidence yet
    assert "variants" in seen[1]                 # round 1: prior variants fed back
