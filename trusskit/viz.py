"""Visualization helpers for inspecting a TrussModel and its AnalysisResult.

All functions accept an optional matplotlib Axes and return it, so they can be
composed.  `inspect_model` builds a 4-panel overview figure and optionally saves it.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401  (registers 3D)

TENSION, COMPRESSION, NEUTRAL = "#c0392b", "#2471a3", "#7f8c8d"
_VIEWS = {"front": (0, 2, "x [mm]", "z [mm]"),   # looking along +y
          "top": (0, 1, "x [mm]", "y [mm]"),
          "side": (1, 2, "y [mm]", "z [mm]")}


def _segments(model, disp=None, scale=1.0):
    segs = {}
    for mid, m in model.members.items():
        a = model.nodes[m.start_node].position()
        b = model.nodes[m.end_node].position()
        if disp is not None:
            a = a + scale * disp[m.start_node]
            b = b + scale * disp[m.end_node]
        segs[mid] = (a, b)
    return segs


def _axes(ax, projection3d):
    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection="3d" if projection3d else None)
    return ax


def _equal_3d(ax, pts):
    """True-proportion 3D axes fitted tightly around the points."""
    pts = np.asarray(pts)
    lo, hi = pts.min(0), pts.max(0)
    ext = np.maximum(hi - lo, 1e-9)
    pad = 0.05 * ext.max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_zlim(lo[2] - pad, hi[2] + pad)
    ax.set_box_aspect(ext + 2 * pad, zoom=0.95)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")


def _force_style(forces, tol_frac=1e-6):
    fmax = max((abs(f) for f in forces.values()), default=1.0) or 1.0
    col, lw = {}, {}
    for mid, f in forces.items():
        if abs(f) <= tol_frac * fmax:
            col[mid], lw[mid] = NEUTRAL, 0.8
        else:
            col[mid] = TENSION if f > 0 else COMPRESSION
            lw[mid] = 0.8 + 4.0 * abs(f) / fmax
    return col, lw


def plot_truss_3d(model, ax=None, label_nodes=False, label_members=False,
                  supports=None, load_case=None, elev=22, azim=-62):
    """Undeformed geometry in 3D; optional ID labels, support and load markers."""
    ax = _axes(ax, True)
    segs = _segments(model)
    for a, b in segs.values():
        ax.plot(*zip(a, b), color="#34495e", lw=1.4)
    P = np.array([n.position() for n in model.nodes.values()])
    ax.scatter(*P.T, s=14, color="black", depthshade=False)
    if label_nodes:
        for n in model.nodes.values():
            ax.text(n.x, n.y, n.z, f" {n.id}", fontsize=7, color="#1e8449")
    if label_members:
        for mid, (a, b) in segs.items():
            mpt = (a + b) / 2
            ax.text(*mpt, f"{mid}", fontsize=6, color="#8e44ad")
    if supports:
        S = np.array([model.nodes[s.node_id].position() for s in supports])
        ax.scatter(*S.T, marker="^", s=70, color="#27ae60", depthshade=False, label="support")
    if load_case:
        span = np.ptp(P, axis=0).max()
        for nid, f in load_case.nodal_forces.items():
            p = model.nodes[nid].position()
            d = np.asarray(f) / (np.linalg.norm(f) or 1) * 0.18 * span
            ax.quiver(*(p - d), *d, color="#e67e22", arrow_length_ratio=0.25, lw=2)
    _equal_3d(ax, P)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(model.name)
    return ax


def plot_forces_3d(model, result, ax=None, elev=22, azim=-62):
    """Member axial forces: red = tension, blue = compression, width ~ |F|."""
    ax = _axes(ax, True)
    col, lw = _force_style(result.member_forces)
    segs = _segments(model)
    for mid, (a, b) in segs.items():
        ax.plot(*zip(a, b), color=col[mid], lw=lw[mid])
    _equal_3d(ax, np.array([n.position() for n in model.nodes.values()]))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"Axial forces – {result.load_case}")
    ax.legend(handles=[Line2D([], [], color=TENSION, label="tension"),
                       Line2D([], [], color=COMPRESSION, label="compression"),
                       Line2D([], [], color=NEUTRAL, label="~zero")], loc="upper left", fontsize=7)
    return ax


def plot_deformed_3d(model, result, scale=None, ax=None, elev=22, azim=-62):
    """Undeformed (grey) vs. deformed (colored by displacement magnitude)."""
    ax = _axes(ax, True)
    P = np.array([n.position() for n in model.nodes.values()])
    umax = max(np.linalg.norm(u) for u in result.displacements.values()) or 1.0
    if scale is None:
        scale = 0.08 * np.ptp(P, axis=0).max() / umax   # exaggerate to ~8% of size
    for a, b in _segments(model).values():
        ax.plot(*zip(a, b), color="#bdc3c7", lw=0.8)
    dsegs = _segments(model, result.displacements, scale)
    cmap = plt.get_cmap("viridis")
    for mid, (a, b) in dsegs.items():
        m = model.members[mid]
        mag = 0.5 * (np.linalg.norm(result.displacements[m.start_node]) +
                     np.linalg.norm(result.displacements[m.end_node])) / umax
        ax.plot(*zip(a, b), color=cmap(mag), lw=1.8)
    D = np.array([m for a, b in dsegs.values() for m in (a, b)])
    _equal_3d(ax, np.vstack([P, D]))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"Deformed ×{scale:.0f} – {result.load_case}\n"
                 f"max |u| = {umax:.3f} mm")
    return ax


def plot_view_2d(model, view="front", result=None, ax=None):
    """Orthographic projection (front=xz, top=xy, side=yz); forces coloring if result."""
    i, j, xl, yl = _VIEWS[view]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.5))
    segs = _segments(model)
    if result is not None:
        col, lw = _force_style(result.member_forces)
    for mid, (a, b) in segs.items():
        c = col[mid] if result else "#34495e"
        w = lw[mid] if result else 1.2
        ax.plot([a[i], b[i]], [a[j], b[j]], color=c, lw=w, alpha=0.85)
    P = np.array([n.position() for n in model.nodes.values()])
    ax.scatter(P[:, i], P[:, j], s=10, color="black", zorder=3)
    ax.set_aspect("equal"); ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(f"{view} view" + (f" – {result.load_case}" if result else ""))
    ax.grid(alpha=0.25)
    return ax


def inspect_model(model, result=None, supports=None, load_case=None, path=None, show=False):
    """4-panel overview: 3D geometry | 3D forces or deformed | front view | top view."""
    fig = plt.figure(figsize=(14, 9))
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    plot_truss_3d(model, ax1, supports=supports, load_case=load_case)
    if result is not None:
        ax2 = fig.add_subplot(2, 2, 2, projection="3d")
        plot_forces_3d(model, result, ax2)
    ax3 = fig.add_subplot(2, 2, 3)
    plot_view_2d(model, "front", result, ax3)
    ax4 = fig.add_subplot(2, 2, 4)
    plot_view_2d(model, "top", result, ax4)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    if show:
        plt.show()
    return fig


def _fs_color(fs, req):
    if fs == float("inf"):
        return NEUTRAL, 0.8
    if fs < 1.0:
        return "#7b241c", 3.2
    if fs < req:
        return "#e74c3c", 3.0
    if fs < 2 * req:
        return "#f1c40f", 2.4
    return "#27ae60", 1.8


def plot_safety_factors(model, fs_by_member, fs_required=1.5, view="3d", ax=None, title=None):
    """Color members by safety factor: dark red <1, red <required, yellow <2x, green OK.
    fs_by_member: {member_id: FS or EnvelopeEntry or MemberFailure}."""
    def val(x):
        return getattr(x, "fs", getattr(x, "governing_fs", x))
    segs = _segments(model)
    P = np.array([n.position() for n in model.nodes.values()])
    if view == "3d":
        ax = _axes(ax, True)
        for mid, (a, b) in segs.items():
            c, w = _fs_color(val(fs_by_member[mid]), fs_required)
            ax.plot(*zip(a, b), color=c, lw=w)
        _equal_3d(ax, P)
        ax.view_init(elev=22, azim=-62)
    else:
        i, j, xl, yl = _VIEWS[view]
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 3.5))
        for mid, (a, b) in segs.items():
            c, w = _fs_color(val(fs_by_member[mid]), fs_required)
            ax.plot([a[i], b[i]], [a[j], b[j]], color=c, lw=w)
            if val(fs_by_member[mid]) < fs_required:
                mp = (a + b) / 2
                ax.annotate(f"{mid}", (mp[i], mp[j]), fontsize=7, ha="center")
        ax.set_aspect("equal"); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(alpha=0.25)
    ax.set_title(title or f"Safety factor (required {fs_required})")
    ax.legend(handles=[Line2D([], [], color="#7b241c", label="FS < 1"),
                       Line2D([], [], color="#e74c3c", label=f"FS < {fs_required}"),
                       Line2D([], [], color="#f1c40f", label=f"FS < {2 * fs_required:g}"),
                       Line2D([], [], color="#27ae60", label="OK"),
                       Line2D([], [], color=NEUTRAL, label="no force")],
              loc="upper left", fontsize=7)
    return ax
