import numpy as np
import pytest
from trusskit import *


def _model(n=4):
    return generate_truss(LayoutParameters(layout="triangular", panels=n))


@pytest.mark.parametrize("n", [2, 4, 6, 8])
def test_counts_and_isostatic(n):
    m = _model(n)
    assert len(m.nodes) == 3 * (n + 1)
    assert len(m.members) == 3 * (n + 1) + 6 * n          # rings + chords + diagonals
    assert len(m.members) == 3 * len(m.nodes) - 6          # exactly isostatic
    assert m.validate()[0] == []


def test_cross_section_geometry():
    m = _model(4)
    lo, hi = m.bounding_box()
    assert np.allclose(lo, [0, -15, 0]) and np.allclose(hi, [220, 15, 40])
    apex = [n for n in m.nodes.values() if n.z == 40]
    assert len(apex) == 5 and all(n.y == 0 for n in apex)
    assert sorted({n.x for n in m.nodes.values()}) == [0, 55, 110, 165, 220]


@pytest.mark.parametrize("cond,total", [("C", 78.48), ("D", 156.96)])
def test_solves_with_correct_reactions(cond, total):
    m = _model(4)
    sup = standard_supports(m)
    res = solve_truss(m, sup, create_load_case(m, cond, sup), SolidCircular(3.0),
                      PLA_PLACEHOLDER)
    assert sum(r[2] for r in res.reactions.values()) == pytest.approx(total)
    assert res.max_vertical_displacement() > 0


def test_lateral_displacement_small_vs_vertical():
    # The single bottom-face diagonal makes the truss slightly asymmetric about y = 0,
    # so a lateral drift (~9% of vertical for the default geometry) is expected.
    m = _model(4)
    sup = standard_supports(m)
    res = solve_truss(m, sup, create_load_case(m, "C", sup), SolidCircular(3.0),
                      PLA_PLACEHOLDER)
    assert res.max_lateral_displacement() < 0.15 * res.max_vertical_displacement()


def test_odd_panels_allowed():
    assert generate_truss(LayoutParameters(layout="triangular", panels=3)).validate()[0] == []