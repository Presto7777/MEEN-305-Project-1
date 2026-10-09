"""Task 5: generalized direct-stiffness solver for 3D pin-jointed trusses.

Units: mm, N, MPa.  Tension positive.  Works on any TrussModel.
"""
from dataclasses import dataclass
import numpy as np
from .geometry import TrussModel


class UnstableStructureError(RuntimeError):
    """Raised when the reduced stiffness matrix is singular / ill-conditioned."""


class EquilibriumError(RuntimeError):
    """Raised when reactions + loads fail the force or moment balance."""


@dataclass
class AnalysisResult:
    load_case: str
    displacements: dict    # node_id -> [ux, uy, uz] (mm)
    reactions: dict        # node_id -> [Rx, Ry, Rz] (N) (zero in free dofs)
    member_forces: dict    # member_id -> axial force (N), + tension
    stiffness_ratio: float  # lambda_min / lambda_max of K_ff (stability indicator)
    force_residual: float
    moment_residual: float

    def max_vertical_displacement(self):
        return max(abs(u[2]) for u in self.displacements.values())

    def max_lateral_displacement(self):
        return max(abs(u[1]) for u in self.displacements.values())


def _section_for(sections, sid):
    if isinstance(sections, dict):
        return sections[sid]
    return sections  # single CrossSection used for all members


def total_mass(model, sections, material):
    """Ideal member mass [g] = sum(rho * A * L).  Excludes joints and print overhead."""
    return sum(material.density * _section_for(sections, m.section_id).area()
               * m.length(model.nodes) for m in model.members.values())


def solve_truss(model: TrussModel, supports, load_case, sections, material,
                stability_tol=1e-9, equilibrium_tol=1e-8) -> AnalysisResult:
    errors, _ = model.validate()
    if errors:
        raise ValueError("invalid model: " + "; ".join(errors))

    idx = model.node_index_map()
    ndof = 3 * len(idx)
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    geom = {}
    for mid, m in model.members.items():
        L = m.length(model.nodes)
        n = m.direction(model.nodes)
        A = _section_for(sections, m.section_id).area()
        q = np.outer(n, n)
        ke = material.E * A / L * np.block([[q, -q], [-q, q]])
        i, j = idx[m.start_node], idx[m.end_node]
        dofs = np.r_[3 * i:3 * i + 3, 3 * j:3 * j + 3]
        K[np.ix_(dofs, dofs)] += ke
        geom[mid] = (L, n, A, i, j)

    for nid, f in load_case.nodal_forces.items():
        if nid not in idx:
            raise ValueError(f"load applied at missing node {nid}")
        F[3 * idx[nid]:3 * idx[nid] + 3] += np.asarray(f, dtype=float)

    fixed = np.zeros(ndof, dtype=bool)
    for s in supports:
        if s.node_id not in idx:
            raise ValueError(f"support at missing node {s.node_id}")
        for a, flag in enumerate(s.fixed_dofs):
            if flag:
                fixed[3 * idx[s.node_id] + a] = True
    free = ~fixed
    if not free.any():
        raise ValueError("all degrees of freedom are constrained")

    Kff = K[np.ix_(free, free)]
    eig = np.linalg.eigvalsh(Kff)
    ratio = float(eig[0] / eig[-1])
    if eig[0] <= stability_tol * eig[-1]:
        raise UnstableStructureError(
            f"reduced stiffness matrix is singular/ill-conditioned "
            f"(lambda_min/lambda_max = {ratio:.2e}): mechanism or insufficient supports")

    u = np.zeros(ndof)
    u[free] = np.linalg.solve(Kff, F[free])   # never invert K explicitly
    if not np.all(np.isfinite(u)):
        raise UnstableStructureError("non-finite displacements")

    R = K @ u - F
    R[free] = 0.0

    # member axial forces
    forces = {}
    for mid, (L, n, A, i, j) in geom.items():
        delta = n @ (u[3 * j:3 * j + 3] - u[3 * i:3 * i + 3])
        forces[mid] = float(material.E * A / L * delta)

    # equilibrium checks
    pos = np.array([model.nodes[nid].position() for nid in sorted(idx, key=idx.get)])
    Fext = F.reshape(-1, 3)
    Rr = R.reshape(-1, 3)
    fres = float(np.linalg.norm(Fext.sum(axis=0) + Rr.sum(axis=0)))
    mres = float(np.linalg.norm(np.cross(pos, Fext).sum(axis=0) +
                                np.cross(pos, Rr).sum(axis=0)))
    fscale = max(np.abs(Fext).sum(), 1e-12)
    mscale = max((np.linalg.norm(pos, axis=1) * np.linalg.norm(Fext, axis=1)).sum(), 1e-12)
    if fres > equilibrium_tol * fscale or mres > equilibrium_tol * mscale:
        raise EquilibriumError(f"force residual {fres:.3e}, moment residual {mres:.3e}")

    inv = {v: k for k, v in idx.items()}
    disp = {inv[k]: u[3 * k:3 * k + 3].copy() for k in range(len(idx))}
    reac = {inv[k]: Rr[k].copy() for k in range(len(idx)) if np.any(fixed[3 * k:3 * k + 3])}
    return AnalysisResult(load_case.name, disp, reac, forces, ratio, fres, mres)
