import json
import matplotlib
matplotlib.use("Agg")
import pytest
from trusskit import *
from trusskit.explorer import Explorer
from trusskit.design_report import analyze_design, compare_designs


def _spec(**kw):
    base = dict(layout="pratt", height=50, depth=35, panels=8, orientation="side", K=0.8,
                name="test", notes="n",
                sections={"chord": {"kind": "hollow rectangular", "params": [6, 6, 1.0]},
                          "web": {"kind": "hollow circular", "params": [5, 0.8]},
                          "brace": {"kind": "rectangular", "params": [2, 3]}})
    base.update(kw)
    return DesignSpec(**base)


def test_roundtrip_dict_and_file(tmp_path):
    s = _spec(material=Material("custom", 3000.0, 1.2e-3, Strength(tensile_inplane=40, knockdown=0.8)))
    assert DesignSpec.from_dict(s.to_dict()) == s
    p = s.save(tmp_path / "sub" / "d.json")
    assert DesignSpec.load(p) == s


def test_results_block_is_informational(tmp_path):
    s = _spec()
    ev = s.evaluate()
    p = s.save(tmp_path / "d.json", ev)
    data = json.load(open(p))
    assert data["results"]["min_fs"] == pytest.approx(ev.min_fs)
    assert DesignSpec.load(p) == s                       # 'results' ignored on load


def test_evaluate_matches_evaluate_design():
    s = _spec()
    a = s.evaluate()
    b = evaluate_design("pratt", 220, 50, 35, 8, orientation="side", K=0.8,
                        sections=s.build_sections())
    assert a.min_fs == pytest.approx(b.min_fs) and a.mass == pytest.approx(b.mass)
    assert set(a.results) == {"C", "D"} and set(a.failures) == {"C", "D"}


def test_reloaded_design_gives_identical_results(tmp_path):
    s = _spec()
    s.save(tmp_path / "d.json")
    a, b = s.evaluate(), DesignSpec.load(tmp_path / "d.json").evaluate()
    assert a.min_fs == b.min_fs and a.mass == b.mass


def test_validation_errors():
    with pytest.raises(ValueError):
        _spec(sections={"chord": {"kind": "hexagon", "params": [1]},
                        "web": {"kind": "solid circular", "params": [1]},
                        "brace": {"kind": "solid circular", "params": [1]}})
    with pytest.raises(ValueError):
        _spec(sections={"chord": {"kind": "rectangular", "params": [1]},
                        "web": {"kind": "solid circular", "params": [1]},
                        "brace": {"kind": "solid circular", "params": [1]}})
    with pytest.raises(ValueError):
        DesignSpec.from_dict({"schema": 99})


def test_defaults_evaluate():
    ev = DesignSpec().evaluate()
    assert ev.mass > 0 and ev.min_fs > 0


def test_explorer_save_and_load_roundtrip(tmp_path):
    ex = Explorer(designs_dir=str(tmp_path))
    ex.radio_layout.set_active(3)                      # triangular
    ex.sliders["height"].set_val(55)
    ex.radio_type.set_active(3)                        # chord -> hollow rectangular
    ex.param_sliders[2].set_val(0.9)
    ex.radio_group.set_active(1); ex.radio_type.set_active(1)    # web -> hollow circular
    ex.param_sliders[1].set_val(0.7)
    path = ex.save_design()
    saved = DesignSpec.load(path)
    assert saved.layout == "triangular" and saved.height == 55
    assert saved.sections["chord"]["kind"] == "hollow rectangular"
    assert saved.sections["chord"]["params"][2] == pytest.approx(0.9)
    assert saved.sections["web"]["params"][1] == pytest.approx(0.7)
    assert saved.evaluate().min_fs == pytest.approx(ex.current.min_fs)
    ex.plt.close("all")

    ex2 = Explorer(designs_dir=str(tmp_path), spec=saved)       # restore into a fresh window
    assert ex2.current_spec == saved
    assert ex2.current.min_fs == pytest.approx(ex.current.min_fs)
    assert ex2.radio_layout.value_selected == "triangular"
    assert ex2.radio_type.value_selected == "hollow rectangular"
    assert len(ex2.param_sliders) == 3 and len(ex2.trace) == 1
    ex2.plt.close("all")


def test_explorer_load_snaps_panels(tmp_path):
    ex = Explorer(designs_dir=str(tmp_path))
    ex.load_spec(DesignSpec(panels=6))
    assert ex.current_spec.panels in (4, 8)            # snapped to a multiple of 4
    assert "snaps" in ex.txt_status.get_text()
    ex.plt.close("all")


def test_explorer_save_trace(tmp_path):
    ex = Explorer(designs_dir=str(tmp_path))
    ex.sliders["height"].set_val(45); ex.sliders["height"].set_val(50)
    n = len(ex.trace)
    path = ex.save_trace()
    data = json.load(open(path))
    assert len(data["designs"]) == n and data["designs"][0]["results"]["mass_g"] > 0
    assert [DesignSpec.from_dict(d) for d in data["designs"]][-1] == ex.current_spec
    rows = open(path[:-5] + ".csv").read().strip().splitlines()
    assert len(rows) == n + 1
    ex.plt.close("all")


def test_analyze_design_writes_outputs(tmp_path):
    text = analyze_design(_spec(), str(tmp_path), K_values=(1.0,), knockdowns=(1.0,))
    for f in ("summary.txt", "design.json", "overview_C.png", "overview_D.png",
              "deformed_C.png", "safety_factors.png", "results_C_nodes.csv",
              "results_D_members.csv"):
        assert (tmp_path / f).exists(), f
    assert "Load margin" in text and "Sensitivity" in text


def test_compare_designs(tmp_path):
    specs = [_spec(name="a"), DesignSpec(name="b")]
    table, evs = compare_designs(specs, path=str(tmp_path / "c.png"))
    assert (tmp_path / "c.png").exists() and len(evs) == 2 and "a" in table and "b" in table