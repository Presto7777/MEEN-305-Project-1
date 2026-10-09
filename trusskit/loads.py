"""Task 4: supports and load cases."""
from dataclasses import dataclass
import numpy as np

G = 9.81  # m/s^2


def mass_to_force(mass_kg: float) -> float:
    """Weight in newtons."""
    return mass_kg * G


@dataclass
class Support:
    node_id: int
    fixed_dofs: tuple  # (ux, uy, uz); True = restrained


@dataclass
class LoadCase:
    name: str
    nodal_forces: dict  # node_id -> np.ndarray([Fx, Fy, Fz])  [N]


def standard_supports(model, tol=1e-6):
    """Supports on the bottom-chord end nodes (lowest-z nodes at min/max x).

    Left end : first node pinned (x,y,z); other left nodes fix x and z.
    Right end: z only (roller).  This kills all six rigid-body modes while
    leaving the structure free to expand.  Replace with the real fixture.
    """
    lo, hi = model.bounding_box()
    bottom = [n for n in model.nodes.values() if abs(n.z - lo[2]) <= tol]
    xmin = min(n.x for n in bottom)
    xmax = max(n.x for n in bottom)
    left = sorted((n for n in bottom if abs(n.x - xmin) <= tol), key=lambda n: n.y)
    right = sorted((n for n in bottom if abs(n.x - xmax) <= tol), key=lambda n: n.y)
    sup = []
    if len(left) == 1:
        sup.append(Support(left[0].id, (True, True, True)))
        sup += [Support(n.id, (False, True, True)) for n in right]
    else:
        sup.append(Support(left[0].id, (True, True, True)))
        sup += [Support(n.id, (True, False, True)) for n in left[1:]]
        sup += [Support(n.id, (False, False, True)) for n in right]
    return sup


def create_load_case(model, condition, supports=None, mass_kg=8.0, tol=1e-6):
    """Condition 'C': one load at mid-span.  'D': two loads at quarter points.

    Loads are applied on the bottom chord; if several nodes share the load
    station (e.g. two side trusses) the force is split equally among them.
    Positions are measured between the support x-coordinates.
    """
    P = mass_to_force(mass_kg)
    if supports:
        xs = [model.nodes[s.node_id].x for s in supports]
        x0, x1 = min(xs), max(xs)
    else:
        lo, hi = model.bounding_box()
        x0, x1 = lo[0], hi[0]
    L = x1 - x0
    zmin = model.bounding_box()[0][2]

    cond = condition.upper()
    if cond == "C":
        stations = [(x0 + 0.5 * L, P)]
    elif cond == "D":
        stations = [(x0 + 0.25 * L, P), (x0 + 0.75 * L, P)]
    else:
        raise ValueError("condition must be 'C' or 'D'")

    forces, total = {}, 0.0
    for x, Pk in stations:
        ids = model.find_nodes(x=x, z=zmin, tol=tol)
        if not ids:
            raise ValueError(f"no bottom-chord node at x={x:.3f} mm; add a load "
                             f"node (e.g. use a panel count divisible by 4)")
        for nid in ids:
            f = np.zeros(3)
            f[2] = -Pk / len(ids)
            forces[nid] = forces.get(nid, np.zeros(3)) + f
        total += Pk
    assert abs(sum(f[2] for f in forces.values()) + total) < 1e-9
    return LoadCase(f"Condition {cond}", forces)
