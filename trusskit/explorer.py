"""Interactive design explorer: min safety factor vs. mass with live sliders.

    python -m trusskit.explorer                 # or: python examples/interactive_explorer.py

evaluate_design() is pure logic (no matplotlib); Explorer adds the matplotlib GUI.
Each member group (chord / web / brace) has its own section TYPE and parameters.
"""
from dataclasses import dataclass
import numpy as np
from .layouts import LayoutParameters, generate_truss
from .groups import assign_groups, GROUPS
from .sections import SolidCircular, SECTION_TYPES, make_section
from .materials import PLA_PLACEHOLDER
from .loads import standard_supports, create_load_case
from .solver import solve_truss, total_mass, UnstableStructureError, EquilibriumError
from .failure import analyze_failures, envelope, check_design

LAYOUTS = ("warren", "pratt", "howe", "triangular")
ORIENTATIONS = ("bottom", "top", "side")
GROUP_BASE_SIZE = {"chord": 4.0, "web": 3.0, "brace": 2.0}   # nominal size for default sections


@dataclass
class DesignEvaluation:
    layout: str
    orientation: str
    panels: int
    section_desc: dict       # group -> text description of its section
    min_features: dict       # group -> smallest wall/dimension [mm]
    group_counts: dict
    model: object
    sections: dict           # group -> CrossSection
    mass: float              # g (ideal members)
    check: object            # DesignCheck (min FS over C and D)
    envelope: dict           # member -> EnvelopeEntry
    max_uz: dict             # {"C": mm, "D": mm}
    max_uy: dict
    fs_required: float

    @property
    def min_fs(self):
        return self.check.min_fs


def evaluate_design(layout, span, height, depth, panels, d_chord=None, d_web=None, d_brace=None,
                    orientation="bottom", K=1.0, material=PLA_PLACEHOLDER,
                    fs_required=1.5, mass_kg=8.0, sections=None) -> DesignEvaluation:
    """Generate, size, solve C and D, and check safety factors.

    Sizing: either give round diameters d_chord/d_web/d_brace, or pass
    sections={"chord": CrossSection, "web": ..., "brace": ...} (takes precedence).
    """
    model = assign_groups(generate_truss(LayoutParameters(
        span=span, height=height, depth=depth, panels=int(panels), layout=layout)))
    if sections is None:
        if None in (d_chord, d_web, d_brace):
            raise ValueError("give d_chord, d_web, d_brace or a sections dict")
        sections = {"chord": SolidCircular(d_chord), "web": SolidCircular(d_web),
                    "brace": SolidCircular(d_brace)}
    missing = [g for g in GROUPS if g not in sections]
    if missing:
        raise ValueError(f"sections missing for groups {missing}")
    sup = standard_supports(model)
    frs, uz, uy = [], {}, {}
    for cond in "CD":
        res = solve_truss(model, sup, create_load_case(model, cond, sup, mass_kg),
                          sections, material)
        frs.append(analyze_failures(model, res, sections, material, orientation, K, fs_required))
        uz[cond], uy[cond] = res.max_vertical_displacement(), res.max_lateral_displacement()
    counts = {g: sum(1 for m in model.members.values() if m.section_id == g) for g in GROUPS}
    return DesignEvaluation(
        layout, orientation, int(panels),
        {g: sections[g].describe() for g in GROUPS},
        {g: sections[g].min_feature() for g in GROUPS},
        counts, model, sections, total_mass(model, sections, material),
        check_design(frs, fs_required), envelope(frs), uz, uy, fs_required)


class Explorer:
    """Matplotlib window.  Geometry sliders (width, height, panels), then choose a member
    group, its section type and that type's parameters.  One live point (mass vs min FS)
    plus a trace of every previous setting."""

    COLORS = {"warren": "#2471a3", "pratt": "#e67e22", "howe": "#8e44ad",
              "triangular": "#16a085"}

    def __init__(self, span=220.0, layout="warren", orientation="bottom", K=1.0,
                 fs_required=1.5, material=PLA_PLACEHOLDER, max_trace=300,
                 min_feature_mm=0.8):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, RadioButtons, Button
        self.plt, self._Slider = plt, Slider
        self.span, self.K, self.fs_req, self.material = span, K, fs_required, material
        self.max_trace, self.min_feature_mm = max_trace, min_feature_mm
        self.trace, self.current, self._busy = [], None, False

        # per-group section state: chosen kind + remembered parameters for every kind
        self.spec = {g: {"kind": "solid circular",
                         "params": {k: t.defaults(GROUP_BASE_SIZE[g])
                                    for k, t in SECTION_TYPES.items()}} for g in GROUPS}
        self.edit_group = "chord"

        self.fig = plt.figure(figsize=(14, 9.5))
        self.fig.suptitle(f"Truss explorer  |  span {span:g} mm, K = {K:g}, "
                          f"loads C and D (8 kg)", fontsize=12)
        self.ax_main = self.fig.add_axes([0.07, 0.50, 0.42, 0.42])
        self.ax_view = self.fig.add_axes([0.55, 0.64, 0.41, 0.28])
        self.ax_info = self.fig.add_axes([0.55, 0.40, 0.41, 0.22]); self.ax_info.axis("off")

        geo = [("depth", "width (y) [mm]", 10, 80, 30, 1, "%d"),
               ("height", "height (z) [mm]", 10, 80, 40, 1, "%d"),
               ("panels", "panels (x4)", 4, 16, 4, 4, "%d")]
        self.sliders = {}
        for k, (key, label, lo, hi, init, step, fmt) in enumerate(geo):
            ax = self.fig.add_axes([0.14, 0.38 - 0.05 * k, 0.33, 0.03])
            self.sliders[key] = Slider(ax, label, lo, hi, valinit=init, valstep=step, valfmt=fmt)
            self.sliders[key].on_changed(self.update)
            self._hide_marker(self.sliders[key])

        self.txt_section = self.fig.text(0.07, 0.225, "", fontsize=10, weight="bold")
        self.param_sliders = []

        ax = self.fig.add_axes([0.55, 0.24, 0.13, 0.13]); ax.set_title("layout", fontsize=9)
        self.radio_layout = RadioButtons(ax, LAYOUTS, active=LAYOUTS.index(layout))
        ax = self.fig.add_axes([0.70, 0.27, 0.10, 0.10]); ax.set_title("print orientation", fontsize=9)
        self.radio_orient = RadioButtons(ax, ORIENTATIONS, active=ORIENTATIONS.index(orientation))
        ax = self.fig.add_axes([0.82, 0.27, 0.10, 0.10]); ax.set_title("edit group", fontsize=9)
        self.radio_group = RadioButtons(ax, GROUPS, active=0)
        types = tuple(SECTION_TYPES)
        ax = self.fig.add_axes([0.55, 0.05, 0.22, 0.14]); ax.set_title("section type (edited group)", fontsize=9)
        self.radio_type = RadioButtons(ax, types, active=0)
        ax = self.fig.add_axes([0.82, 0.05, 0.10, 0.05])
        self.btn_clear = Button(ax, "clear trace")

        self.radio_layout.on_clicked(self.update)
        self.radio_orient.on_clicked(self.update)
        self.radio_group.on_clicked(self._group_changed)
        self.radio_type.on_clicked(self._type_changed)
        self.btn_clear.on_clicked(self.clear_trace)
        self.fig.text(0.07, 0.012, "Panels snap to multiples of 4 (condition D needs load nodes at the "
                      "quarter points). Pratt/Howe need even panel counts. Hollow walls t apply on all "
                      "sides (2t <= smallest outer dimension).", fontsize=8, color="#555")
        self._build_param_sliders()
        self.update()

    # ------------------------------------------------------------------ section controls
    @staticmethod
    def _hide_marker(slider):
        try:
            slider.vline.set_visible(False)
        except AttributeError:
            pass

    def _build_param_sliders(self):
        for s in self.param_sliders:
            s.disconnect_events()
            self.fig.delaxes(s.ax)
        self.param_sliders = []
        g = self.edit_group
        kind = self.spec[g]["kind"]
        vals = self.spec[g]["params"][kind]
        self.txt_section.set_text(f"Cross-section: {g}  ->  {kind}")
        for k, (label, lo, hi, step) in enumerate(SECTION_TYPES[kind].params):
            ax = self.fig.add_axes([0.14, 0.17 - 0.05 * k, 0.33, 0.03])
            sl = self._Slider(ax, f"{g}: {label}", lo, hi, valinit=min(max(vals[k], lo), hi),
                              valstep=step, valfmt="%.1f")
            self._hide_marker(sl)
            sl.on_changed(lambda v, k=k, g=g, kind=kind: self._param_changed(g, kind, k, v))
            self.param_sliders.append(sl)

    def _param_changed(self, g, kind, k, val):
        self.spec[g]["params"][kind][k] = float(val)
        self.update()

    def _group_changed(self, label):
        self.edit_group = label
        self._busy = True
        try:
            self.radio_type.set_active(list(SECTION_TYPES).index(self.spec[label]["kind"]))
        finally:
            self._busy = False
        self._build_param_sliders()
        self.fig.canvas.draw_idle()

    def _type_changed(self, label):
        if self._busy:
            return
        self.spec[self.edit_group]["kind"] = label
        self._build_param_sliders()
        self.update()

    def current_sections(self):
        return {g: make_section(self.spec[g]["kind"], self.spec[g]["params"][self.spec[g]["kind"]])
                for g in GROUPS}

    # ------------------------------------------------------------------ evaluation
    def evaluate(self):
        s = self.sliders
        return evaluate_design(self.radio_layout.value_selected, self.span, s["height"].val,
                               s["depth"].val, s["panels"].val,
                               orientation=self.radio_orient.value_selected, K=self.K,
                               material=self.material, fs_required=self.fs_req,
                               sections=self.current_sections())

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
        xmax = max(m.max() * 1.3, ev.mass * 1.3, 5.0)
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
        lines = [
            f"{ev.layout} | print: {ev.orientation} | K = {self.K:g} | panels {ev.panels} | "
            f"{len(ev.model.nodes)} nodes, {len(ev.model.members)} members"]
        for g in GROUPS:
            lines.append(f"  {g:<5} x{ev.group_counts[g]:<3} {ev.section_desc[g]}")
        lines += [
            f"mass {ev.mass:.2f} g   min FS {c.min_fs:.2f}  -> {'PASS' if c.passes else 'FAIL'}",
            f"governing: member {c.member_id}, {c.mode}, {c.load_case}; "
            f"{len(c.failing)} member(s) below {self.fs_req:g}",
            f"max uz  C {ev.max_uz['C']:.2f} mm  D {ev.max_uz['D']:.2f} mm;  "
            f"max |uy|  C {ev.max_uy['C']:.3f}  D {ev.max_uy['D']:.3f} mm"]
        thin = [g for g in GROUPS if ev.min_features[g] < self.min_feature_mm]
        if thin:
            lines.append(f"PRINT WARNING: {', '.join(thin)} feature < {self.min_feature_mm:g} mm "
                         f"(about two 0.4 mm perimeters)")
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