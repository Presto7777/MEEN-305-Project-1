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
    thick = evaluate_design(layout, 220, 40, 30, 4, 8, 7, 4)
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


def test_evaluate_with_mixed_section_types():
    secs = {"chord": make_section("hollow rectangular", [6, 6, 1.2]),
            "web": make_section("hollow circular", [5, 0.8]),
            "brace": make_section("rectangular", [2, 2])}
    ev = evaluate_design("warren", 220, 40, 30, 4, sections=secs)
    assert ev.mass > 0 and ev.min_fs > 0
    assert "rect tube" in ev.section_desc["chord"] and "tube" in ev.section_desc["web"]
    sup = standard_supports(ev.model)
    frs = [analyze_failures(ev.model, solve_truss(ev.model, sup, create_load_case(ev.model, c, sup),
                                                  secs, PLA_PLACEHOLDER), secs, PLA_PLACEHOLDER)
           for c in "CD"]
    assert ev.min_fs == pytest.approx(check_design(frs).min_fs)
    assert ev.mass == pytest.approx(total_mass(ev.model, secs, PLA_PLACEHOLDER))


def test_evaluate_requires_sizing():
    with pytest.raises(ValueError):
        evaluate_design("warren", 220, 40, 30, 4)


def test_explorer_group_and_type_controls():
    ex = Explorer()
    assert len(ex.param_sliders) == 1                        # solid circular: one parameter
    n0 = len(ex.trace)
    ex.radio_type.set_active(3)                              # chord -> hollow rectangular
    assert len(ex.param_sliders) == 3 and len(ex.trace) == n0 + 1
    assert "rect tube" in ex.current.section_desc["chord"]
    ex.radio_group.set_active(1)                             # edit web (still solid circular)
    assert ex.edit_group == "web" and len(ex.param_sliders) == 1
    assert ex.radio_type.value_selected == "solid circular"
    ex.radio_type.set_active(1)                              # web -> hollow circular (2 params)
    assert len(ex.param_sliders) == 2
    ex.param_sliders[1].set_val(1.2)                         # wall t
    assert "tube" in ex.current.section_desc["web"]
    ex.radio_group.set_active(0)                             # back to chord: type radio follows
    assert ex.radio_type.value_selected == "hollow rectangular"
    ex.plt.close("all")


def test_explorer_invalid_section_does_not_crash_or_add_trace():
    ex = Explorer()
    ex.radio_type.set_active(3)                              # chord hollow rect: b=h=6, t=1
    n = len(ex.trace)
    ex.param_sliders[2].set_val(5.0)                         # t=5 -> 2t=10 > 6 -> invalid
    assert len(ex.trace) == n
    ex.param_sliders[2].set_val(1.0)                         # valid again
    assert len(ex.trace) == n + 1
    ex.plt.close("all")