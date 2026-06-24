from __future__ import annotations

from dataclasses import dataclass

from edgevla.compression.exit_criteria import (
    diminishing_returns_exit,
    success_exit,
)
from edgevla.compression.pareto import ParetoPoint, hypervolume, pareto_frontier
from edgevla.jetson import GateVerdict


@dataclass
class Variant:
    point: ParetoPoint
    verdict: GateVerdict
    success: float


@dataclass
class LoopResult:
    best: ParetoPoint | None
    frontier: list[ParetoPoint]
    hv_history: list[float]
    rounds: int
    exit_reason: str


def _best_on_frontier(frontier: list[ParetoPoint]) -> ParetoPoint | None:
    if not frontier:
        return None
    return max(frontier, key=lambda p: p.success)


def compression_loop(
    *,
    design,
    sweep,
    baseline_success: float,
    ref_size_mb: float,
    ref_success: float = 0.0,
    max_rounds: int = 10,
    retain: float = 0.90,
    rel_threshold: float = 0.02,
    window: int = 2,
) -> LoopResult:
    frontier: list[ParetoPoint] = []
    hv_history: list[float] = []
    evidence: dict | None = None

    for round_idx in range(max_rounds):
        candidates = design(round_idx, evidence)
        variants = sweep(candidates)

        frontier = pareto_frontier(frontier + [v.point for v in variants])
        hv_history.append(hypervolume(frontier, ref_size_mb, ref_success))

        for v in variants:
            if success_exit(v.verdict, v.success, baseline_success, retain):
                return LoopResult(
                    best=v.point,
                    frontier=frontier,
                    hv_history=hv_history,
                    rounds=round_idx + 1,
                    exit_reason="success",
                )

        if diminishing_returns_exit(hv_history, rel_threshold, window):
            return LoopResult(
                best=_best_on_frontier(frontier),
                frontier=frontier,
                hv_history=hv_history,
                rounds=round_idx + 1,
                exit_reason="diminishing_returns",
            )

        evidence = {"variants": variants, "frontier": frontier}

    return LoopResult(
        best=_best_on_frontier(frontier),
        frontier=frontier,
        hv_history=hv_history,
        rounds=max_rounds,
        exit_reason="max_rounds",
    )
