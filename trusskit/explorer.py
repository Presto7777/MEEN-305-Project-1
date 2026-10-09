
"""Interactive design explorer: min safety factor vs. mass with live sliders.

    python -m trusskit.explorer                 # or: python examples/interactive_explorer.py

evaluate_design() is pure logic (no matplotlib); Explorer adds the matplotlib GUI.
"""
from dataclasses import dataclass
import numpy as np
from .layouts import LayoutParameters, generate_truss
from .groups import assign_groups, GROUPS
from .sections import SolidCircular
from .materials import PLA_PLACEHOLDER
from .loads import standard_supports, create_load_case
from .solver import solve_truss, total_mass, UnstableStructureError, EquilibriumError
from .failure import analyze_failures, envelope, check_design

LAYOUTS = ("warren", "pratt", "howe", "triangular")
ORIENTATIONS = ("bottom", "top", "side")


@dataclass
class DesignEvaluation:
    layout: str
    orientation: str
    panels: int
    diameters: dict          # {"chord": d, "web": d, "brace": d}
    group_counts: dict
    model: object
    sections: dict
    mass: float              # g (ideal members)
    check: object            # DesignCheck (min FS over C and D)
    envelope: dict           # member -> EnvelopeEntry
    max_uz: dict             # {"C": mm, "D": mm}
    max_uy: dict
    fs_required: float

    @property
    def min_fs(self):
        return self.check.min_fs


def evaluate_design(layout, span, height, depth, panels, d_chord, d_web, d_brace,
                    orientation="bottom", K=1.0, material=PLA_PLACEHOLDER,
                    fs_required=1.5, mass_kg=8.0) -> DesignEvaluation:
    """Generate, size (three diameter groups), solve C and D, and check safety factors."""
    model = assign_groups(generate_truss(LayoutParameters(
        span=span, height=height, depth=depth, panels=int(panels), layout=layout)))
    diam = {"chord": d_chord, "web": d_web, "brace": d_brace}
    sections = {g: SolidCircular(d) for g, d in diam.items()}
    sup = standard_supports(model)
    frs, uz, uy = [], {}, {}
    for cond in "CD":
        res = solve_truss(model, sup, create_load_case(model, cond, sup, mass_kg),
                          sections, material)
        frs.append(analyze_failures(model, res, sections, material, orientation, K, fs_required))
        uz[cond], uy[cond] = res.max_vertical_displacement(), res.max_lateral_displacement()
    counts = {g: sum(1 for m in model.members.values() if m.section_id == g) for g in GROUPS}
    return DesignEvaluation(layout, orientation, int(panels), diam, counts, model, sections,
                            total_mass(model, sections, material), check_design(frs, fs_required),
                            envelope(frs), uz, uy, fs_required)


class Explorer:
    """Matplotlib window: sliders for geometry and group diameters; one live point
    (mass vs min FS) plus a trace of every previous setting."""

    COLORS = {"warren": "#2471a3", "pratt": "#e67e22", "howe": "#8e44ad",
              "triangular": "#16a085"}

    def __init__(self, span=220.0, layout="warren", orientation="bottom", K=1.0,
                 fs_required=1.5, material=PLA_PLACEHOLDER, max_trace=300):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, RadioButtons, Button
        self.plt = plt
        self.span, self.K, self.fs_req, self.material = span, K, fs_required, material
        self.max_trace = max_trace
        self.trace = []          # (mass, min_fs, layout)
        self.current = None

        self.fig = plt.figure(figsize=(14, 8.5))
        self.fig.suptitle(f"Truss explorer  |  span {span:g} mm, K = {K:g}, "
                          f"loads C and D (8 kg)", fontsize=12)
        self.ax_main = self.fig.add_axes([0.07, 0.47, 0.42, 0.45])
        self.ax_view = self.fig.add_axes([0.55, 0.57, 0.41, 0.33])
        self.ax_info = self.fig.add_axes([0.55, 0.40, 0.41, 0.15]); self.ax_info.axis("off")

        specs = [  # key, label, min, max, init, step, fmt
            ("depth", "width (y) [mm]", 10, 80, 30, 1, "%d"),
            ("height", "height (z) [mm]", 10, 80, 40, 1, "%d"),
            ("panels", "panels (x4)", 4, 16, 4, 4, "%d"),
            ("chord", "chord Ø [mm]", 1, 10, 4, 0.1, "%.1f"),
            ("web", "web Ø [mm]", 1, 10, 3, 0.1, "%.1f"),
            ("brace", "brace Ø [mm]", 1, 10, 2, 0.1, "%.1f"),
        ]
        self.sliders = {}
        for k, (key, label, lo, hi, init, step, fmt) in enumerate(specs):
            ax = self.fig.add_axes([0.14, 0.31 - 0.052 * k, 0.33, 0.03])
            self.sliders[key] = Slider(ax, label, lo, hi, valinit=init, valstep=step, valfmt=fmt)
            self.sliders[key].on_changed(self.update)

        ax = self.fig.add_axes([0.55, 0.06, 0.16, 0.26]); ax.set_title("layout", fontsize=9)
        self.radio_layout = RadioButtons(ax, LAYOUTS, active=LAYOUTS.index(layout))
        ax = self.fig.add_axes([0.73, 0.14, 0.12, 0.18]); ax.set_title("print orientation", fontsize=9)
        self.radio_orient = RadioButtons(ax, ORIENTATIONS, active=ORIENTATIONS.index(orientation))
        ax = self.fig.add_axes([0.87, 0.26, 0.09, 0.05])
        self.btn_clear = Button(ax, "clear trace")
        self.radio_layout.on_clicked(self.update)
        self.radio_orient.on_clicked(self.update)
        self.btn_clear.on_clicked(self.clear_trace)
        self.fig.text(0.07, 0.015, "Panels snap to multiples of 4 (condition D needs load nodes at the "
                      "quarter points). Pratt/Howe need even panel counts.", fontsize=8, color="#555")
        self.update()

    # ------------------------------------------------------------------ callbacks
    def evaluate(self):
        s = self.sliders
        return evaluate_design(self.radio_layout.value_selected, self.span, s["height"].val,
                               s["depth"].val, s["panels"].val, s["chord"].val, s["web"].val,
                               s["brace"].val, self.radio_orient.value_selected, self.K,
                               self.material, self.fs_req)

    def update(self, _=None):
        try:
            ev = self.evaluate()
        except (ValueError, UnstableStructureError, EquilibriumError) as exc:
            self._draw_error(str(exc)); return
        self.current = ev
        self.trace.append((ev.mass, ev.min_fs, ev.layout))
        self.trace = self.trace[-self.max_trace:]
        self._draw(ev)

    def clear_trace(self, _=None):
        self.trace = []
        if self.current is not None:
            self.trace.append((self.current.mass, self.current.min_fs, self.current.layout))
            self._draw(self.current)

    # ------------------------------------------------------------------ drawing
    def _draw(self, ev):
        from matplotlib.ticker import FuncFormatter
        from . import viz
        ax = self.ax_main; ax.clear()
        m = np.array([t[0] for t in self.trace]); f = np.array([t[1] for t in self.trace])
        ymax = max(10.0, 2.0 * f.max(), 2.0 * ev.min_fs)
        ymin = min(0.2, 0.7 * f.min())
        xmax = max(m.max() * 1.15, ev.mass * 1.15, 5.0)
        ax.axhspan(self.fs_req, ymax, color="#27ae60", alpha=0.10)
        ax.axhline(self.fs_req, color="#27ae60", lw=1.5, ls="--")
        ax.text(0.99, self.fs_req * 1.07, f"required FS = {self.fs_req:g}", ha="right",
                transform=ax.get_yaxis_transform(), fontsize=8, color="#1e8449")
        ax.plot(m, f, "-", color="#999", lw=0.8, alpha=0.6, zorder=1)
        for lay, col in self.COLORS.items():
            sel = [i for i, t in enumerate(self.trace) if t[2] == lay]
            if sel:
                ax.scatter(m[sel], f[sel], s=22, color=col, alpha=0.55, label=lay, zorder=2)
        ok = ev.check.passes
        ax.scatter([ev.mass], [ev.min_fs], s=170, marker="o", zorder=4, linewidths=2,
                   facecolor="#27ae60" if ok else "#e74c3c", edgecolor="black")
        ax.annotate(f" {ev.mass:.1f} g, FS {ev.min_fs:.2f}", (ev.mass, ev.min_fs),
                    textcoords="offset points", xytext=(10, 8), fontsize=9, weight="bold")
        ax.set_yscale("log"); ax.set_xlim(0, xmax); ax.set_ylim(ymin, ymax)
        ax.set_yticks([v for v in (0.1, 0.2, 0.5, 1, 1.5, 2, 5, 10, 20, 50, 100, 200)
                       if ymin <= v <= ymax])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.set_xlabel("ideal member mass [g]"); ax.set_ylabel("minimum safety factor (log)")
        ax.set_title("Min safety factor vs. mass", fontsize=11)
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="lower right", fontsize=8, title="trace", title_fontsize=8)

        self.ax_view.clear()
        viz.plot_safety_factors(ev.model, ev.envelope, self.fs_req, "front", self.ax_view,
                                title=f"{ev.layout}: worst FS per member (front view)")
        self._draw_info(ev)
        self.fig.canvas.draw_idle()

    def _draw_info(self, ev):
        c = ev.check
        d = ev.diameters
        lines = [
            f"{ev.layout} | print: {ev.orientation} | K = {self.K:g} | panels {ev.panels} | "
            f"{len(ev.model.nodes)} nodes, {len(ev.model.members)} members",
            f"members: chord {ev.group_counts['chord']} (Ø{d['chord']:.1f}), "
            f"web {ev.group_counts['web']} (Ø{d['web']:.1f}), "
            f"brace {ev.group_counts['brace']} (Ø{d['brace']:.1f})",
            f"mass {ev.mass:.2f} g   min FS {c.min_fs:.2f}  -> {'PASS' if c.passes else 'FAIL'}",
            f"governing: member {c.member_id}, {c.mode}, {c.load_case}; "
            f"{len(c.failing)} member(s) below {self.fs_req:g}",
            f"max uz  C {ev.max_uz['C']:.2f} mm  D {ev.max_uz['D']:.2f} mm;  "
            f"max |uy|  C {ev.max_uy['C']:.3f}  D {ev.max_uy['D']:.3f} mm",
        ]
        self.ax_info.clear(); self.ax_info.axis("off")
        self.ax_info.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.5,
                          color="#1e8449" if c.passes else "#a93226")

    def _draw_error(self, msg):
        self.ax_info.clear(); self.ax_info.axis("off")
        self.ax_info.text(0, 1, f"Cannot evaluate this design:\n{msg}", va="top", fontsize=9,
                          color="#a93226", wrap=True)
        self.fig.canvas.draw_idle()


def launch(**kwargs):
    ex = Explorer(**kwargs)
    ex.plt.show()
    return ex


if __name__ == "__main__":
    launch()
