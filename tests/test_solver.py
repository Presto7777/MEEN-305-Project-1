import numpy as np
import pytest
from trusskit import (Node, Member, TrussModel, Support, LoadCase, SolidCircular,
                      Material, PLA_PLACEHOLDER, LayoutParameters, generate_truss,
                      standard_supports, create_load_case, solve_truss,
                      UnstableStructureError)

MAT = Material("test", E=1000.0, density=1e-3)


def test_A_axial_bar():
    L, P = 100.0, 50.0
    sec = SolidCircular(4.0)
    model = TrussModel({0: Node(0, 0, 0, 0), 1: Node(1, L, 0, 0)}, {0: Member(0, 0, 1)})
    sup = [Support(0, (True, True, True)), Support(1, (False, True, True))]
    res = solve_truss(model, sup, LoadCase("axial", {1: np.array([P, 0, 0])}), sec, MAT)
    assert res.displacements[1][0] == pytest.approx(P * L / (MAT.E * sec.area()))
    assert res.member_forces[0] == pytest.approx(P)          # tension
    assert res.reactions[0][0] == pytest.approx(-P)


def test_B_symmetric_triangle():
    a, h, P = 100.0, 60.0, 80.0
    nodes = {0: Node(0, 0, 0, 0), 1: Node(1, 2 * a, 0, 0), 2: Node(2, a, 0, h)}
    members = {0: Member(0, 0, 1), 1: Member(1, 0, 2), 2: Member(2, 1, 2)}
    sup = [Support(0, (True, True, True)), Support(1, (False, True, True)),
           Support(2, (False, True, False))]
    res = solve_truss(TrussModel(nodes, members), sup,
                      LoadCase("apex", {2: np.array([0, 0, -P])}), SolidCircular(5), MAT)
    sin_t = h / np.hypot(a, h)
    assert res.member_forces[1] == pytest.approx(-P / (2 * sin_t))   # compression
    assert res.member_forces[2] == pytest.approx(-P / (2 * sin_t))
    assert res.member_forces[0] == pytest.approx(P * a / (2 * h))    # tension
    assert res.reactions[0][2] == pytest.approx(P / 2)


def _bridge(layout="warren", cond="C", scale=1.0):
    model = generate_truss(LayoutParameters(layout=layout))
    sup = standard_supports(model)
    lc = create_load_case(model, cond, sup)
    for k in lc.nodal_forces:
        lc.nodal_forces[k] = lc.nodal_forces[k] * scale
    return model, sup, lc


@pytest.mark.parametrize("layout", ["warren", "pratt", "howe"])
@pytest.mark.parametrize("cond,total", [("C", 78.48), ("D", 156.96)])
def test_C_equilibrium_and_reactions(layout, cond, total):
    model, sup, lc = _bridge(layout, cond)
    res = solve_truss(model, sup, lc, SolidCircular(3.0), PLA_PLACEHOLDER)
    assert sum(r[2] for r in res.reactions.values()) == pytest.approx(total)
    assert res.force_residual < 1e-6 and res.moment_residual < 1e-4
    left = sum(r[2] for nid, r in res.reactions.items() if model.nodes[nid].x == 0)
    assert left == pytest.approx(total / 2)      # symmetric -> half each end
    assert res.max_vertical_displacement() > 0


def test_D_linear_scaling():
    m, s, lc1 = _bridge(scale=1.0)
    _, _, lc2 = _bridge(scale=2.0)
    sec = SolidCircular(3.0)
    r1 = solve_truss(m, s, lc1, sec, PLA_PLACEHOLDER)
    r2 = solve_truss(m, s, lc2, sec, PLA_PLACEHOLDER)
    for mid in r1.member_forces:
        assert r2.member_forces[mid] == pytest.approx(2 * r1.member_forces[mid], abs=1e-9)
    for nid in r1.displacements:
        assert np.allclose(r2.displacements[nid], 2 * r1.displacements[nid], atol=1e-12)


def test_E_mechanism_detected():
    nodes = {i: Node(i, 10.0 * i, 0, 0) for i in range(3)}      # collinear chain
    members = {0: Member(0, 0, 1), 1: Member(1, 1, 2)}
    sup = [Support(0, (True, True, True)), Support(2, (True, True, True))]
    with pytest.raises(UnstableStructureError):
        solve_truss(TrussModel(nodes, members), sup,
                    LoadCase("z", {1: np.array([0, 0, -1.0])}), SolidCircular(2), MAT)


def test_E_missing_supports_detected():
    model, _, lc = _bridge()
    with pytest.raises(UnstableStructureError):
        solve_truss(model, [Support(0, (True, True, True))], lc,
                    SolidCircular(3.0), PLA_PLACEHOLDER)


def test_load_node_missing_gives_clear_error():
    model = generate_truss(LayoutParameters(panels=6, layout="warren"))
    with pytest.raises(ValueError, match="load node"):
        create_load_case(model, "D", standard_supports(model))
