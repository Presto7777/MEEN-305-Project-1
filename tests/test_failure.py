import math
import numpy as np
import pytest
from trusskit import *

MAT = PLA_PLACEHOLDER
S = MAT.strength


def bar(axis, L, load, d, orientation="bottom"):
    """Single axis-aligned bar along unit vector `axis`, axial `load` at the free end."""
    ax = np.array(axis, float)
    model = TrussModel({0: Node(0, 0, 0, 0), 1: Node(1, *(ax * L))}, {0: Member(0, 0, 1)})
    sup = [Support(0, (True, True, True)),
           Support(1, tuple(bool(c == 0) for c in ax))]
    sec = SolidCircular(d)
    res = solve_truss(model, sup, LoadCase("bar", {1: ax * load}), sec, MAT)
    return analyze_failures(model, res, sec, MAT, orientation), sec


def test_euler_buckling_bar():
    L, d, P = 100.0, 4.0, 20.0
    fr, sec = bar([1, 0, 0], L, -P, d)
    Pcr = math.pi ** 2 * MAT.E * sec.I_min() / L ** 2
    m = fr.members[0]
    assert m.regime == "euler"
    assert m.fs["buckling"] == pytest.approx(Pcr / P)
    assert m.governing_mode == "buckling"


def test_tension_inplane():
    fr, sec = bar([1, 0, 0], 100.0, 300.0, 4.0)
    m = fr.members[0]
    assert m.fs["tension"] == pytest.approx(S.tensile_inplane / (300 / sec.area()))
    assert "interlayer tension" not in m.fs and "buckling" not in m.fs


def test_vertical_member_interlayer_tension_governs_bottom_print():
    fr, sec = bar([0, 0, 1], 100.0, 100.0, 4.0, "bottom")
    m = fr.members[0]
    sigma = 100 / sec.area()
    assert m.governing_mode == "interlayer tension"
    assert m.governing_fs == pytest.approx(S.interlayer_tension / sigma)
    # same member printed on its side lies in the layer plane -> in-layer strength
    fr2, _ = bar([0, 0, 1], 100.0, 100.0, 4.0, "side")
    assert fr2.members[0].governing_mode == "tension"
    assert fr2.members[0].governing_fs == pytest.approx(S.tensile_inplane / sigma)


def test_layer_tractions_45_degrees():
    sn, tau = layer_tractions([1, 0, 1], [0, 0, 1], 10.0)
    assert sn == pytest.approx(5.0) and tau == pytest.approx(5.0)
    assert layer_tractions([1, 0, 0], [0, 0, 1], 10.0) == pytest.approx((0.0, 0.0))
    sn, tau = layer_tractions([0, 0, 1], [0, 0, 1], 10.0)
    assert sn == pytest.approx(10.0) and tau == pytest.approx(0.0)


def test_johnson_for_stocky_member():
    fr, sec = bar([1, 0, 0], 10.0, -50.0, 4.0)       # KL/r = 10 << lambda_c
    m = fr.members[0]
    assert m.regime == "johnson" and any("stocky" in f for f in m.flags)
    Pcr_euler = math.pi ** 2 * MAT.E * sec.I_min() / 10.0 ** 2
    assert m.fs["buckling"] < Pcr_euler / 50.0       # Johnson < Euler here


def test_zero_force_member_not_divided():
    model = generate_truss(LayoutParameters(layout="pratt"))
    sup = standard_supports(model)
    sec = SolidCircular(3.0)
    res = solve_truss(model, sup, create_load_case(model, "D", sup), sec, MAT)
    fr = analyze_failures(model, res, sec, MAT)
    zero = [m for m in fr.members.values() if m.state == "zero"]
    assert zero and all(m.governing_fs == float("inf") and m.governing_mode == "n/a"
                        for m in zero)


def test_fs_halves_when_load_doubles():
    model = generate_truss(LayoutParameters())
    sup = standard_supports(model)
    sec = SolidCircular(3.0)
    lc = create_load_case(model, "C", sup)
    lc2 = LoadCase("2x", {k: 2 * v for k, v in lc.nodal_forces.items()})
    f1 = analyze_failures(model, solve_truss(model, sup, lc, sec, MAT), sec, MAT)
    f2 = analyze_failures(model, solve_truss(model, sup, lc2, sec, MAT), sec, MAT)
    for mid, m in f1.members.items():
        if m.governing_fs != float("inf") and m.governing_fs < 1e6:
            assert f2.members[mid].governing_fs == pytest.approx(m.governing_fs / 2, rel=1e-6)


def _check(layout, d, orientation="bottom"):
    model = generate_truss(LayoutParameters(layout=layout))
    sup = standard_supports(model)
    sec = SolidCircular(d)
    frs = [analyze_failures(model, solve_truss(model, sup, create_load_case(model, c, sup),
                                               sec, MAT), sec, MAT, orientation)
           for c in "CD"]
    return check_design(frs), frs, model


@pytest.mark.parametrize("layout", ["warren", "pratt", "howe", "triangular"])
def test_thin_fails_thick_passes(layout):
    thin, _, _ = _check(layout, 3.0)
    thick, _, _ = _check(layout, 8.0)
    assert not thin.passes and thin.failing
    assert thick.passes and thick.min_fs > thin.min_fs


def test_envelope_takes_minimum_over_cases():
    _, frs, _ = _check("warren", 3.0)
    env = envelope(frs)
    for mid, e in env.items():
        assert e.fs == min(fr.members[mid].governing_fs for fr in frs)


@pytest.mark.parametrize("o", ["bottom", "top", "side"])
def test_orientations_run(o):
    chk, frs, model = _check("warren", 4.0, o)
    assert math.isfinite(chk.min_fs)
    assert "FS" in failure_table(model, frs[0], top=3)


def test_bad_orientation_name():
    with pytest.raises(ValueError):
        PrintOrientation.from_name("diagonal")


def test_knockdown_scales_strength_modes():
    soft = Material("soft", 3500.0, 1.24e-3, Strength(knockdown=0.5))
    model = TrussModel({0: Node(0, 0, 0, 0), 1: Node(1, 100, 0, 0)}, {0: Member(0, 0, 1)})
    sup = [Support(0, (True, True, True)), Support(1, (False, True, True))]
    sec = SolidCircular(4.0)
    res = solve_truss(model, sup, LoadCase("t", {1: np.array([100.0, 0, 0])}), sec, soft)
    fr = analyze_failures(model, res, sec, soft)
    assert fr.members[0].fs["tension"] == pytest.approx(0.5 * 50.0 / (100 / sec.area()))