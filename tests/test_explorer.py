
import matplotlib
matplotlib.use("Agg")
import pytest
from trusskit import *
from trusskit.explorer import Explorer, LAYOUTS


def test_assign_groups_triangular_counts():
    m = assign_groups(generate_truss(LayoutParameters(layout="triangular", panels=4)))
    c = {g: sum(1 for x in m.members.values() if x.section_id == g)
         for g in ("chord", "web", "brace")}
    assert c == {"chord": 12, "brace": 9, "web": 18}


def test_assign_groups_does_not_mutate_input():
    m0 = generate_truss(LayoutParameters())
    assign_groups(m0)
    assert all(x.section_id == "default" for x in m0.members.values())


def test_chords_are_x_aligned():
    m = assign_groups(generate_truss(LayoutParameters(layout="pratt")))
    for x in m.members.values():
        if x.section_id == "chord":
            assert abs(x.direction(m.nodes)[0]) > 0.99


def test_evaluate_matches_manual_pipeline():
    ev = evaluate_design("warren", 220, 40, 30, 4, 3.0, 3.0, 3.0)
    model = generate_truss(LayoutParameters(layout="warren"))
    sup = standard_supports(model)
    sec = SolidCircular(3.0)
    frs = [analyze_failures(model, solve_truss(model, sup, create_load_case(model, c, sup), sec,
                                               PLA_PLACEHOLDER), sec, PLA_PLACEHOLDER)
           for c in "CD"]
    assert ev.min_fs == pytest.approx(check_design(frs).min_fs)
    assert ev.mass == pytest.approx(total_mass(model, sec, PLA_PLACEHOLDER))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_bigger_sections_raise_fs_and_mass(layout):
    thin = evaluate_design(layout, 220, 40, 30, 4, 3, 2, 1.5)
    thick = evaluate_design(layout, 220, 40, 30, 4, 6, 4, 3)
    assert thick.min_fs > thin.min_fs and thick.mass > thin.mass
    assert thick.check.passes


def test_explorer_gui_updates_trace():
    ex = Explorer()
    n0 = len(ex.trace)
    ex.sliders["height"].set_val(60)
    ex.sliders["panels"].set_val(8)
    ex.radio_layout.set_active(3)          # triangular
    ex.radio_orient.set_active(2)          # side
    assert len(ex.trace) == n0 + 4
    assert ex.current.layout == "triangular" and ex.current.panels == 8
    ex.clear_trace()
    assert len(ex.trace) == 1
    ex.plt.close("all")


def test_explorer_survives_invalid_combo():
    ex = Explorer(layout="pratt")
    ex.sliders["panels"].set_val(8)        # fine
    ex.update()
    ex.plt.close("all")