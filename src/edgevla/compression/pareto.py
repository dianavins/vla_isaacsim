from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParetoPoint:
    size_mb: float
    success: float
    label: str


def dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
    no_worse = a.size_mb <= b.size_mb and a.success >= b.success
    strictly_better = a.size_mb < b.size_mb or a.success > b.success
    return no_worse and strictly_better


def pareto_frontier(points: list[ParetoPoint]) -> list[ParetoPoint]:
    front = [
        p for p in points
        if not any(dominates(q, p) for q in points if q is not p)
    ]
    return sorted(front, key=lambda p: p.size_mb)


def hypervolume(
    frontier: list[ParetoPoint],
    ref_size_mb: float,
    ref_success: float,
) -> float:
    """2D hypervolume: minimize size, maximize success.

    Sum of rectangles from each frontier point toward the worst-corner
    reference (ref_size_mb, ref_success). Frontier sorted by ascending size;
    each point covers width (ref_size_mb - size) over the success band above
    the previous (lower-success) point.
    """
    relevant = sorted(
        (p for p in frontier if p.size_mb <= ref_size_mb and p.success >= ref_success),
        key=lambda p: p.success,
    )
    hv = 0.0
    prev_success = ref_success
    for p in relevant:
        width = ref_size_mb - p.size_mb
        height = p.success - prev_success
        hv += width * height
        prev_success = p.success
    return hv
