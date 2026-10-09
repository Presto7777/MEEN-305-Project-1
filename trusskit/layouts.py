"""Task 3: parametric layout generator (Warren / Pratt / Howe, 3D box).

Generates geometry only.  Each layout is a pair of vertical side trusses at
y = -depth/2 and +depth/2, joined by transverse members, bottom- and top-plane
diagonal bracing, and transverse bracing at the two end bays (these stop
cross-section racking).  Intermediate transverse frames can be added later.

Warren : top nodes staggered at panel mid-points (true triangulated web).
Pratt / Howe : aligned top/bottom stations, verticals at every station,
               diagonals leaning toward (Pratt) or away from (Howe) mid-span.
               Require an even panel count.
"""
from dataclasses import dataclass
from .geometry import Node, Member, TrussModel


@dataclass
class LayoutParameters:
    span: float = 220.0
    height: float = 40.0
    depth: float = 30.0
    panels: int = 8
    layout: str = "warren"
    section_id: str = "default"


def generate_truss(p: LayoutParameters) -> TrussModel:
    layout = p.layout.lower()
    n = p.panels
    if layout not in ("warren", "pratt", "howe"):
        raise ValueError(f"unknown layout '{p.layout}'")
    if n < 2:
        raise ValueError("need at least 2 panels")
    if layout in ("pratt", "howe") and n % 2:
        raise ValueError("Pratt/Howe layouts need an even number of panels")

    xb = [p.span * i / n for i in range(n + 1)]
    xt = [p.span * (i + 0.5) / n for i in range(n)] if layout == "warren" else list(xb)
    ys = (-p.depth / 2.0, p.depth / 2.0)

    nodes, key_id = {}, {}
    for s, y in enumerate(ys):
        for lvl, xs, z in (("b", xb, 0.0), ("t", xt, p.height)):
            for i, x in enumerate(xs):
                nid = len(nodes)
                nodes[nid] = Node(nid, x, y, z)
                key_id[(s, lvl, i)] = nid

    edges = []

    def e(a, b):
        edges.append((key_id[a], key_id[b]))

    nt = len(xt)
    for s in (0, 1):
        for i in range(n):
            e((s, "b", i), (s, "b", i + 1))
        for i in range(nt - 1):
            e((s, "t", i), (s, "t", i + 1))
        if layout == "warren":
            for i in range(n):
                e((s, "b", i), (s, "t", i))
                e((s, "t", i), (s, "b", i + 1))
        else:
            for i in range(n + 1):
                e((s, "b", i), (s, "t", i))
            for i in range(n):
                left = i < n // 2
                lean_in = (left == (layout == "pratt"))
                if lean_in:   # top(i) -> bottom(i+1)
                    e((s, "t", i), (s, "b", i + 1))
                else:         # bottom(i) -> top(i+1)
                    e((s, "b", i), (s, "t", i + 1))

    # transverse members
    for i in range(n + 1):
        e((0, "b", i), (1, "b", i))
    for i in range(nt):
        e((0, "t", i), (1, "t", i))
    # plane bracing (alternating diagonals) in bottom and top chord planes
    for lvl, cnt in (("b", n), ("t", nt - 1)):
        for i in range(cnt):
            if i % 2 == 0:
                e((0, lvl, i), (1, lvl, i + 1))
            else:
                e((1, lvl, i), (0, lvl, i + 1))
    # transverse bracing at the end bays
    if layout == "warren":
        e((0, "b", 0), (1, "t", 0)); e((1, "b", 0), (0, "t", 0))
        e((0, "b", n), (1, "t", nt - 1)); e((1, "b", n), (0, "t", nt - 1))
    else:
        e((0, "b", 0), (1, "t", 0))
        e((0, "b", n), (1, "t", n))

    members = {i: Member(i, a, b, p.section_id) for i, (a, b) in enumerate(edges)}
    model = TrussModel(nodes, members,
                       name=f"{layout}_n{n}_h{p.height:g}_d{p.depth:g}")
    errors, _ = model.validate()
    if errors:
        raise ValueError("generated invalid truss: " + "; ".join(errors))
    return model
