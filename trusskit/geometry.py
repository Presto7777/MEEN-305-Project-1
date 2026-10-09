"""Task 1 & 2: Node, Member, TrussModel and validation.

Geometry only: no forces, materials or section properties live here.
Coordinate system: x along span, y across depth, z vertical. Units: mm.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np


@dataclass(frozen=True)
class Node:
    id: int
    x: float
    y: float
    z: float

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass(frozen=True)
class Member:
    id: int
    start_node: int
    end_node: int
    section_id: str = "default"

    def length(self, nodes: dict) -> float:
        d = nodes[self.end_node].position() - nodes[self.start_node].position()
        return float(np.linalg.norm(d))

    def direction(self, nodes: dict) -> np.ndarray:
        d = nodes[self.end_node].position() - nodes[self.start_node].position()
        L = np.linalg.norm(d)
        if L == 0.0:
            raise ValueError(f"member {self.id} has zero length")
        return d / L


@dataclass
class TrussModel:
    nodes: dict
    members: dict
    name: str = "truss"
    _build_errors: list = field(default_factory=list, repr=False)

    @classmethod
    def from_lists(cls, node_list, member_list, name="truss"):
        """Build from iterables so duplicate IDs can be detected by validate()."""
        errs, nodes, members = [], {}, {}
        for n in node_list:
            if n.id in nodes:
                errs.append(f"duplicate node id {n.id}")
            nodes[n.id] = n
        for m in member_list:
            if m.id in members:
                errs.append(f"duplicate member id {m.id}")
            members[m.id] = m
        return cls(nodes, members, name, errs)

    # ---------- queries ----------
    def member_lengths(self) -> dict:
        return {mid: m.length(self.nodes) for mid, m in self.members.items()}

    def bounding_box(self):
        P = np.array([n.position() for n in self.nodes.values()])
        return P.min(axis=0), P.max(axis=0)

    def node_index_map(self) -> dict:
        """Node ID -> contiguous zero-based matrix index (sorted by ID)."""
        return {nid: i for i, nid in enumerate(sorted(self.nodes))}

    def find_nodes(self, x=None, y=None, z=None, tol=1e-6) -> list:
        out = []
        for n in self.nodes.values():
            if ((x is None or abs(n.x - x) <= tol) and
                    (y is None or abs(n.y - y) <= tol) and
                    (z is None or abs(n.z - z) <= tol)):
                out.append(n.id)
        return sorted(out)

    def connected_components(self) -> list:
        parent = {nid: nid for nid in self.nodes}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for m in self.members.values():
            if m.start_node in parent and m.end_node in parent:
                ra, rb = find(m.start_node), find(m.end_node)
                if ra != rb:
                    parent[ra] = rb
        groups = {}
        for nid in self.nodes:
            groups.setdefault(find(nid), set()).add(nid)
        return list(groups.values())

    # ---------- validation ----------
    def validate(self, tol=1e-9, span_limits=None, max_extent=None):
        """Return (errors, warnings).  Errors must block the solver.

        span_limits: optional (lo, hi) for the x-extent of the model (warning).
        max_extent : optional (x, y, z) build-volume limits (error).
        Connectivity does NOT prove stability: the solver checks that.
        """
        errors = list(self._build_errors)
        warnings = []

        for key, n in self.nodes.items():
            if key != n.id:
                errors.append(f"node key {key} != node.id {n.id}")
            if not all(math.isfinite(v) for v in (n.x, n.y, n.z)):
                errors.append(f"node {key} has non-finite coordinates")
        for key, m in self.members.items():
            if key != m.id:
                errors.append(f"member key {key} != member.id {m.id}")

        seen_pairs = {}
        for mid, m in self.members.items():
            a, b = m.start_node, m.end_node
            ok = True
            for end in (a, b):
                if end not in self.nodes:
                    errors.append(f"member {mid} references missing node {end}")
                    ok = False
            if a == b:
                errors.append(f"member {mid} connects node {a} to itself")
                continue
            pair = frozenset((a, b))
            if pair in seen_pairs:
                errors.append(f"member {mid} duplicates member {seen_pairs[pair]} "
                              f"(nodes {a}-{b})")
            else:
                seen_pairs[pair] = mid
            if ok:
                pa, pb = self.nodes[a].position(), self.nodes[b].position()
                if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                    if np.linalg.norm(pb - pa) <= tol:
                        errors.append(f"member {mid} has (near) zero length")

        # coincident nodes (warning)
        pts = {}
        for n in self.nodes.values():
            pts.setdefault(tuple(np.round(n.position(), 6)), []).append(n.id)
        for p, ids in pts.items():
            if len(ids) > 1:
                warnings.append(f"nodes {ids} are coincident at {p}")

        if self.nodes:
            comps = self.connected_components()
            if len(comps) > 1:
                errors.append(f"model has {len(comps)} disconnected components")
            finite = all(np.all(np.isfinite(n.position())) for n in self.nodes.values())
            if finite:
                lo, hi = self.bounding_box()
                ext = hi - lo
                if span_limits is not None and not (span_limits[0] <= ext[0] <= span_limits[1]):
                    warnings.append(f"x-extent {ext[0]:.2f} mm outside {span_limits}")
                if max_extent is not None:
                    for ax, e, lim in zip("xyz", ext, max_extent):
                        if e > lim:
                            errors.append(f"{ax}-extent {e:.2f} mm exceeds limit {lim}")
        else:
            errors.append("model has no nodes")
        return errors, warnings
