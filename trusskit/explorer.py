"""Interactive design explorer: min safety factor vs. mass with live sliders.

    python -m trusskit.explorer                 # or: python examples/interactive_explorer.py

Each member group (chord / web / brace) has its own section TYPE and parameters.
Designs can be saved to JSON (DesignSpec) and reloaded (--load, Explorer.load_spec).
"""
import csv
import json
import numpy as np
from .groups import GROUPS
from .sections import SECTION_TYPES
from .materials import PLA_PLACEHOLDER
from .solver import UnstableStructureError, EquilibriumError
from .design import (DesignSpec, DesignEvaluation, evaluate_design, timestamped_path,  # noqa: F401
                     LAYOUTS, ORIENTATIONS, GROUP_BASE_SIZE, SCHEMA)


class Explorer:
    """Matplotlib window.  Geometry sliders (width, height, panels), then choose a member
    group, its section type and that type's parameters.  One live point (mass vs min FS)
    plus a trace of every previous setting.  'save design' / 'save trace' write JSON
    (and CSV for the trace) into `designs_dir`; `spec=` / load_spec() restores a design."""

    COLORS = {"warren": "#2471a3", "pratt": "#e67e22", "howe": "#8e44ad",
              "triangular": "#16a085"}

    def __init__(self, span=220.0, layout="warren", orientation="bottom", K=1.0,
                 fs_required=1.5, material=PLA_PLACEHOLDER, max_trace=300,
                 min_feature_mm=0.8, spec=None, designs_dir="designs"):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, RadioButtons, Button
        self.plt, self._Slider = plt, Slider
        self.span, self.K, self.fs_req, self.material = span, K, fs_required, material
        self.max_trace, self.min_feature_mm = max_trace, min_feature_mm
        self.designs_dir = designs_dir
        self.trace, self.current, self.current_spec = [], None, None
        self._busy = False       # ignore radio callbacks fired by programmatic set_active
        self._suspend = False    # skip re-evaluation during bulk changes

        # per-group section state: chosen kind + remembered parameters for every kind
        self.spec = {g: {"kind": "solid circular",
                         "params": {k: t.defaults(GROUP_BASE_SIZE[g])
                                    for k, t in SECTION_TYPES.items()}} for g in GROUPS}
        self.edit_group = "chord"

        self.fig = plt.figure(figsize=(14, 9.5))
        self.title = self.fig.suptitle("", fontsize=12)
        self._set_title()
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
        self.txt_status = self.fig.text(0.07, 0.035, "", fontsize=8.5, color="#1f618d")
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
        ax = self.fig.add_axes([0.82, 0.19, 0.12, 0.05]); self.btn_save = Button(ax, "save design")
        ax = self.fig.add_axes([0.82, 0.12, 0.12, 0.05]); self.btn_save_trace = Button(ax, "save trace")
        ax = self.fig.add_axes([0.82, 0.05, 0.12, 0.05]); self.btn_clear = Button(ax, "clear trace")

        self.radio_layout.on_clicked(self.update)
        self.radio_orient.on_clicked(self.update)
        self.radio_group.on_clicked(self._group_changed)
        self.radio_type.on_clicked(self._type_changed)
        self.btn_save.on_clicked(self.save_design)
        self.btn_save_trace.on_clicked(self.save_trace)
        self.btn_clear.on_clicked(self.clear_trace)
        self.fig.text(0.07, 0.012, "Panels snap to multiples of 4 (condition D needs load nodes at the "
                      "quarter points). Pratt/Howe need even panel counts. Hollow walls t apply on all "
                      "sides (2t <= smallest outer dimension).", fontsize=8, color="#555")
        self._build_param_sliders()
        if spec is not None:
            self.load_spec(spec)
        else:
            self.update()

    def _set_title(self):
        self.title.set_text(f"Truss explorer  |  span {self.span:g} mm, K = {self.K:g}, "
                            f"loads C and D")

    def _status(self, msg, color="#1f618d"):
        self.txt_status.set_text(msg); self.txt_status.set_color(color)
        self.fig.canvas.draw_idle()

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
        if self._busy:
            return
        self.edit_group = label
        self._sync_type_radio()
        self._build_param_sliders()
        self.fig.canvas.draw_idle()

    def _sync_type_radio(self):
        self._busy = True
        try:
            self.radio_type.set_active(list(SECTION_TYPES).index(self.spec[self.edit_group]["kind"]))
        finally:
            self._busy = False

    def _type_changed(self, label):
        if self._busy:
            return
        self.spec[self.edit_group]["kind"] = label
        self._build_param_sliders()
        self.update()

    # ------------------------------------------------------------------ spec in / out
    def current_design_spec(self) -> DesignSpec:
        s = self.sliders
        return DesignSpec(
            layout=self.radio_layout.value_selected, span=self.span,
            height=float(s["height"].val), depth=float(s["depth"].val),
            panels=int(round(s["panels"].val)),
            sections={g: {"kind": self.spec[g]["kind"],
                          "params": [float(x) for x in self.spec[g]["params"][self.spec[g]["kind"]]]}
                      for g in GROUPS},
            orientation=self.radio_orient.value_selected, K=self.K,
            fs_required=self.fs_req, material=self.material)

    def load_spec(self, spec: DesignSpec):
        """Restore a saved design into the sliders/radios (panels snap to 4, 8, 12, 16)."""
        self.span, self.K, self.fs_req, self.material = spec.span, spec.K, spec.fs_required, spec.material
        self._set_title()
        snapped = int(min(16, max(4, 4 * round(spec.panels / 4))))
        self._suspend = True
        try:
            self.sliders["depth"].set_val(spec.depth)
            self.sliders["height"].set_val(spec.height)
            self.sliders["panels"].set_val(snapped)
            self.radio_layout.set_active(LAYOUTS.index(spec.layout))
            self.radio_orient.set_active(ORIENTATIONS.index(spec.orientation))
            for g in GROUPS:
                kind = spec.sections[g]["kind"]
                self.spec[g]["kind"] = kind
                self.spec[g]["params"][kind] = [float(x) for x in spec.sections[g]["params"]]
            self.edit_group = "chord"
            self._busy = True
            try:
                self.radio_group.set_active(0)
            finally:
                self._busy = False
            self._sync_type_radio()
            self._build_param_sliders()
        finally:
            self._suspend = False
        self.update()
        if snapped != spec.panels:
            self._status(f"loaded '{spec.name or 'design'}' (panels {spec.panels} -> {snapped}: "
                         f"explorer snaps to multiples of 4)", "#b9770e")
        else:
            self._status(f"loaded '{spec.name or 'design'}'")

    def save_design(self, _=None):
        """Write the CURRENT design (+ its results summary) to designs_dir as JSON."""
        if self.current is None:
            self._status("nothing to save yet", "#a93226"); return None
        sp = self.current_spec
        path = timestamped_path(self.designs_dir, f"{sp.layout}_p{sp.panels}")
        sp.save(path, self.current)
        self._status(f"saved design -> {path}")
        return path

    def save_trace(self, _=None):
        """Write every traced design to designs_dir as one JSON list plus a CSV summary."""
        if not self.trace:
            self._status("trace is empty", "#a93226"); return None
        path = timestamped_path(self.designs_dir, "trace")
        entries = []
        for i, (_, _, _, sp, summ) in enumerate(self.trace):
            d = sp.to_dict(); d["index"] = i; d["results"] = summ
            entries.append(d)
        with open(path, "w") as fh:
            json.dump({"schema": SCHEMA, "designs": entries}, fh, indent=2)
        csv_path = path[:-5] + ".csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["index", "layout", "panels", "height", "depth", "orientation", "chord",
                        "web", "brace", "mass_g", "min_fs", "passes", "governing_member",
                        "governing_mode", "governing_case"])
            for i, (_, _, _, sp, sm) in enumerate(self.trace):
                w.writerow([i, sp.layout, sp.panels, sp.height, sp.depth, sp.orientation,
                            sm["section_desc"]["chord"], sm["section_desc"]["web"],
                            sm["section_desc"]["brace"], f"{sm['mass_g']:.4f}",
                            f"{sm['min_fs']:.4f}", sm["passes"], sm["governing_member"],
                            sm["governing_mode"], sm["governing_case"]])
        self._status(f"saved {len(entries)} designs -> {path} (+ .csv)")
        return path

    # ------------------------------------------------------------------ evaluation
    def update(self, _=None):
        if self._suspend:
            return
        try:
            sp = self.current_design_spec()
            ev = sp.evaluate()
        except (ValueError, UnstableStructureError, EquilibriumError) as exc:
            self._draw_error(str(exc)); return
        self.current, self.current_spec = ev, sp
        self.trace.append((ev.mass, ev.min_fs, ev.layout, sp, ev.summary()))
        self.trace = self.trace[-self.max_trace:]
        self._draw(ev)

    def clear_trace(self, _=None):
        self.trace = []
        if self.current is not None:
            self.trace.append((self.current.mass, self.current.min_fs, self.current.layout,
                               self.current_spec, self.current.summary()))
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