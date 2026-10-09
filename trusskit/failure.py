"""Task 6 & 7: stress, buckling, print-orientation checks and safety factors.

Consumes an AnalysisResult (axial forces) and returns per-member safety factors.
Checks, per member and load case (only the applicable ones are reported):
  tension / compression  : |sigma| vs in-layer strength
  buckling (compression) : Euler for slender members, Johnson parabola for stocky
                           ones, effective-length factor K (default 1.0)
  interlayer tension     : sigma_n = sigma (n.l)^2   vs interlayer tensile strength
  interlayer shear       : tau = |sigma||n.l|sqrt(1-(n.l)^2) vs interlayer shear strength
The interlayer checks are an initial orientation-risk estimate (uniaxial stress
resolved onto the layer plane).  They ignore joints, porosity and stress
concentrations -- confirm with coupon and load tests.
"""
from dataclasses import dataclass, field
import math
import numpy as np
from .solver import _section_for

INF = float("inf")

# Layer-plane normal (global coordinates) for each build orientation.
#  bottom : truss printed upright, bottom chord on the bed  -> layers stack in z
#  top    : truss printed upside-down, top chord on the bed -> layers stack in z
#  side   : truss printed lying on one side truss           -> layers stack in y
# 'top' and 'bottom' have the same layer normal (they differ in overhangs/support
# material, which this module does not model).
_ORIENTATIONS = {"bottom": (0.0, 0.0, 1.0), "top": (0.0, 0.0, 1.0),
                 "side": (0.0, 1.0, 0.0), "side_left": (0.0, 1.0, 0.0),
                 "side_right": (0.0, 1.0, 0.0)}


@dataclass(frozen=True)
class PrintOrientation:
    name: str
    layer_normal: tuple

    @classmethod
    def from_name(cls, name):
        try:
            return cls(name, _ORIENTATIONS[name.lower()])
        except KeyError:
            raise ValueError(f"unknown orientation '{name}'; choose from "
                             f"{sorted(_ORIENTATIONS)} or pass a PrintOrientation")


def _orient_for(orientation, mid):
    if isinstance(orientation, dict):      # per-member override, with 'default' key
        orientation = orientation.get(mid, orientation.get("default"))
        if orientation is None:
            raise ValueError(f"no orientation given for member {mid} and no 'default'")
    if isinstance(orientation, PrintOrientation):
        return orientation
    return PrintOrientation.from_name(orientation)


def layer_tractions(axis, layer_normal, sigma):
    """Resolve uniaxial axial stress onto the layer plane.
    Returns (sigma_n, |tau|) = (sigma c^2, |sigma| |c| sqrt(1-c^2)), c = axis . normal."""
    c = float(np.dot(axis, layer_normal) / (np.linalg.norm(axis) * np.linalg.norm(layer_normal)))
    c = max(-1.0, min(1.0, c))
    return sigma * c * c, abs(sigma) * abs(c) * math.sqrt(max(0.0, 1.0 - c * c))


def critical_buckling_load(E, I, A, L, K, Sc):
    """Return (P_cr [N], regime, slenderness).  Euler if KL/r >= lambda_c, else Johnson."""
    r = math.sqrt(I / A)
    lam = K * L / r
    lam_c = math.pi * math.sqrt(2.0 * E / Sc)
    if lam >= lam_c:
        return math.pi ** 2 * E * I / (K * L) ** 2, "euler", lam
    sig = Sc * (1.0 - Sc * lam ** 2 / (4.0 * math.pi ** 2 * E))
    return sig * A, "johnson", lam


@dataclass
class MemberFailure:
    member_id: int
    force: float
    stress: float
    state: str                 # tension | compression | zero
    length: float
    slenderness: float
    regime: str                # euler | johnson | -
    layer_cos: float
    sigma_n: float
    tau: float
    fs: dict                   # applicable failure mode -> safety factor
    governing_fs: float
    governing_mode: str
    flags: list = field(default_factory=list)


@dataclass
class FailureResult:
    load_case: str
    orientation: str
    K: float
    fs_required: float
    members: dict

    def governing_member(self):
        return min(self.members.values(), key=lambda m: m.governing_fs)

    @property
    def min_fs(self):
        return self.governing_member().governing_fs

    @property
    def passes(self):
        return self.min_fs >= self.fs_required


def analyze_failures(model, result, sections, material, orientation="bottom",
                     K=1.0, fs_required=1.5) -> FailureResult:
    s = material.strength
    kd = s.knockdown
    St, Sc = s.tensile_inplane * kd, s.compressive_inplane * kd
    Szt, Szs = s.interlayer_tension * kd, s.interlayer_shear * kd
    fmax = max((abs(f) for f in result.member_forces.values()), default=0.0)
    zero_tol = max(1e-6 * fmax, 1e-9)

    out = {}
    for mid, m in model.members.items():
        F = result.member_forces[mid]
        sec = _section_for(sections, m.section_id)
        A, I = sec.area(), sec.I_min()
        L = m.length(model.nodes)
        n = m.direction(model.nodes)
        orient = _orient_for(orientation, mid)
        l = np.asarray(orient.layer_normal, float)
        cosl = float(np.dot(n, l) / np.linalg.norm(l))
        r = math.sqrt(I / A)
        if abs(F) <= zero_tol:
            out[mid] = MemberFailure(mid, F, 0.0, "zero", L, K * L / r, "-", cosl, 0.0, 0.0,
                                     {}, INF, "n/a")
            continue

        sigma = F / A
        sn, tau = layer_tractions(n, l, sigma)
        fs, flags, regime, lam = {}, [], "-", K * L / r
        if sigma > 0:
            state = "tension"
            fs["tension"] = St / sigma
        else:
            state = "compression"
            fs["compression"] = Sc / abs(sigma)
            Pcr, regime, lam = critical_buckling_load(material.E, I, A, L, K, Sc)
            fs["buckling"] = Pcr / abs(F)
            if regime == "johnson":
                flags.append("stocky: Johnson used (Euler unconservative)")
            if lam > 150:
                flags.append("very slender: imperfection-sensitive")
        if sn > 1e-9 * abs(sigma):
            fs["interlayer tension"] = Szt / sn
        if tau > 1e-9 * abs(sigma):
            fs["interlayer shear"] = Szs / tau
        mode = min(fs, key=fs.get)
        out[mid] = MemberFailure(mid, F, sigma, state, L, lam, regime, cosl, sn, tau,
                                 fs, fs[mode], mode, flags)
    return FailureResult(result.load_case, orient.name if not isinstance(orientation, dict)
                         else "mixed", K, fs_required, out)


# ---------------------------------------------------------------- envelope / design check
@dataclass
class EnvelopeEntry:
    member_id: int
    fs: float
    mode: str
    load_case: str


def envelope(failures) -> dict:
    """Worst (minimum) safety factor per member over several FailureResults."""
    env = {}
    for fr in failures:
        for mid, mf in fr.members.items():
            if mid not in env or mf.governing_fs < env[mid].fs:
                env[mid] = EnvelopeEntry(mid, mf.governing_fs, mf.governing_mode, fr.load_case)
    return env


@dataclass
class DesignCheck:
    min_fs: float
    member_id: int
    mode: str
    load_case: str
    fs_required: float
    passes: bool
    failing: list          # member ids with FS < required

    def __str__(self):
        v = "PASS" if self.passes else "FAIL"
        return (f"{v}: min FS = {self.min_fs:.2f} (required {self.fs_required}) "
                f"member {self.member_id}, {self.mode}, {self.load_case}; "
                f"{len(self.failing)} member(s) below requirement")


def check_design(failures, fs_required=None) -> DesignCheck:
    failures = list(failures)
    req = fs_required if fs_required is not None else failures[0].fs_required
    env = envelope(failures)
    worst = min(env.values(), key=lambda e: e.fs)
    failing = sorted(e.member_id for e in env.values() if e.fs < req)
    return DesignCheck(worst.fs, worst.member_id, worst.mode, worst.load_case, req,
                       worst.fs >= req, failing)


# ---------------------------------------------------------------- text output
def _fmt(v):
    return "   inf" if v == INF else f"{v:6.2f}"


def failure_table(model, failure: FailureResult, top=None) -> str:
    """Members sorted by governing FS (lowest first)."""
    rows = sorted(failure.members.values(), key=lambda m: m.governing_fs)
    if top:
        rows = rows[:top]
    out = [f"{failure.load_case} | orientation: {failure.orientation} | K = {failure.K} | "
           f"required FS = {failure.fs_required}",
           f"{'id':>4} {'F[N]':>8} {'sigma':>7} {'KL/r':>6} {'state':>11} "
           f"{'FS':>6} {'governing':>19}  status"]
    for m in rows:
        status = "ok" if m.governing_fs >= failure.fs_required else "FAIL"
        out.append(f"{m.member_id:>4} {m.force:8.2f} {m.stress:7.2f} {m.slenderness:6.1f} "
                   f"{m.state:>11} {_fmt(m.governing_fs)} {m.governing_mode:>19}  {status}"
                   + (f"  [{'; '.join(m.flags)}]" if m.flags else ""))
    return "\n".join(out)


def failure_summary(failure: FailureResult) -> str:
    g = failure.governing_member()
    n_fail = sum(1 for m in failure.members.values() if m.governing_fs < failure.fs_required)
    modes = {}
    for m in failure.members.values():
        if m.governing_mode != "n/a":
            modes[m.governing_mode] = modes.get(m.governing_mode, 0) + 1
    ms = ", ".join(f"{k}: {v}" for k, v in sorted(modes.items()))
    return "\n".join([
        f"Failure check: {failure.load_case} ({failure.orientation}, K={failure.K})",
        f"  minimum FS {g.governing_fs:.2f} at member {g.member_id} ({g.governing_mode})",
        f"  members below FS {failure.fs_required}: {n_fail} of {len(failure.members)}",
        f"  governing modes: {ms}"])
