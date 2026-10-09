import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from trusskit import *
from trusskit import viz


def _case():
    m = generate_truss(LayoutParameters(layout="pratt"))
    s = standard_supports(m); lc = create_load_case(m, "D", s)
    return m, s, lc, solve_truss(m, s, lc, SolidCircular(3), PLA_PLACEHOLDER)


def test_plots_run(tmp_path):
    m, s, lc, r = _case()
    fig = viz.inspect_model(m, r, s, lc, path=str(tmp_path / "o.png"))
    assert (tmp_path / "o.png").exists()
    plt.close("all")


def test_reports_and_csv(tmp_path):
    m, s, lc, r = _case()
    assert "nodes: 20" in model_summary(m, SolidCircular(3), PLA_PLACEHOLDER)
    assert "Condition D" in result_summary(m, r)
    assert member_table(m, r, SolidCircular(3), top=3).count("\n") == 3
    export_csv(m, r, str(tmp_path / "x"), SolidCircular(3))
    assert (tmp_path / "x_nodes.csv").exists() and (tmp_path / "x_members.csv").exists()
