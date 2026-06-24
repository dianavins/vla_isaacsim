from edgevla.compression.pareto import (
    ParetoPoint,
    dominates,
    hypervolume,
    pareto_frontier,
)


def test_dominance():
    a = ParetoPoint(size_mb=200, success=0.85, label="a")
    b = ParetoPoint(size_mb=240, success=0.75, label="b")
    assert dominates(a, b) is True
    assert dominates(b, a) is False


def test_frontier_drops_dominated_points():
    pts = [
        ParetoPoint(900, 0.88, "orig"),
        ParetoPoint(450, 0.87, "int8"),
        ParetoPoint(230, 0.82, "int4"),
        ParetoPoint(240, 0.75, "int4+badprune"),  # dominated by int4
    ]
    front = pareto_frontier(pts)
    labels = [p.label for p in front]
    assert "int4+badprune" not in labels
    assert labels == ["int4", "int8", "orig"]  # sorted by size ascending


def test_hypervolume_grows_when_frontier_improves():
    ref_size, ref_success = 1000.0, 0.0
    small = [ParetoPoint(500, 0.80, "a")]
    bigger = [ParetoPoint(500, 0.80, "a"), ParetoPoint(250, 0.82, "b")]
    hv_small = hypervolume(small, ref_size, ref_success)
    hv_big = hypervolume(bigger, ref_size, ref_success)
    assert hv_big > hv_small
