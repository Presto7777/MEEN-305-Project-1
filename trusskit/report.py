"""Text and CSV output for inspecting a model / analysis result."""
import csv
import numpy as np
from .solver import total_mass, _section_for


def model_summary(model, sections=None, material=None) -> str:
    errors, warnings = model.validate()
    lo, hi = model.bounding_box()
    L = model.member_lengths()
    lines = [f"Model: {model.name}",
             f"  nodes: {len(model.nodes)}   members: {len(model.members)}   "
             f"components: {len(model.connected_components())}",
             f"  bounding box [mm]: x {lo[0]:.1f}..{hi[0]:.1f}, y {lo[1]:.1f}..{hi[1]:.1f}, "
             f"z {lo[2]:.1f}..{hi[2]:.1f}",
             f"  member length [mm]: min {min(L.values()):.2f}  max {max(L.values()):.2f}  "
             f"total {sum(L.values()):.1f}",
             f"  DOFs: {3 * len(model.nodes)}"]
    if sections is not None and material is not None:
        lines.append(f"  ideal member mass: {total_mass(model, sections, material):.2f} g "
                     f"({material.name})")
    lines.append(f"  validation: {len(errors)} errors, {len(warnings)} warnings")
    lines += [f"    ERROR: {e}" for e in errors] + [f"    warn:  {w}" for w in warnings]
    return "\n".join(lines)


def result_summary(model, result) -> str:
    f = result.member_forces
    t = max(f, key=f.get); c = min(f, key=f.get)
    uz = min(result.displacements.items(), key=lambda kv: kv[1][2])
    uy = max(result.displacements.items(), key=lambda kv: abs(kv[1][1]))
    rz = sum(r[2] for r in result.reactions.values())
    zero = sum(1 for v in f.values() if abs(v) < 1e-6 * max(abs(x) for x in f.values()))
    return "\n".join([
        f"Result: {result.load_case}",
        f"  max tension     : member {t:>3}  {f[t]:+9.2f} N",
        f"  max compression : member {c:>3}  {f[c]:+9.2f} N",
        f"  near-zero members: {zero} of {len(f)}",
        f"  max downward uz : node {uz[0]:>3}  {uz[1][2]:+9.4f} mm",
        f"  max lateral |uy|: node {uy[0]:>3}  {uy[1][1]:+9.5f} mm",
        f"  sum of reactions Rz: {rz:.4f} N",
        f"  equilibrium residuals: force {result.force_residual:.2e} N, "
        f"moment {result.moment_residual:.2e} N*mm",
        f"  stiffness ratio lambda_min/lambda_max: {result.stiffness_ratio:.2e}"])


def member_table(model, result, sections=None, top=None) -> str:
    """Per-member table sorted by |force| (largest first)."""
    L = model.member_lengths()
    rows = sorted(result.member_forces.items(), key=lambda kv: -abs(kv[1]))
    if top:
        rows = rows[:top]
    out = [f"{'id':>4} {'nodes':>9} {'L[mm]':>8} {'F[N]':>9} {'state':>11}"
           + (f" {'sigma[MPa]':>11}" if sections is not None else "")]
    fmax = max(abs(v) for v in result.member_forces.values()) or 1
    for mid, F in rows:
        m = model.members[mid]
        state = "~zero" if abs(F) < 1e-6 * fmax else ("tension" if F > 0 else "compression")
        s = f"{mid:>4} {m.start_node:>4}-{m.end_node:<4} {L[mid]:8.2f} {F:9.2f} {state:>11}"
        if sections is not None:
            s += f" {F / _section_for(sections, m.section_id).area():11.2f}"
        out.append(s)
    return "\n".join(out)


def export_csv(model, result, prefix, sections=None):
    """Write <prefix>_nodes.csv, <prefix>_members.csv for spreadsheet inspection."""
    with open(f"{prefix}_nodes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "x", "y", "z", "ux", "uy", "uz", "Rx", "Ry", "Rz"])
        for nid, n in sorted(model.nodes.items()):
            u = result.displacements[nid]
            r = result.reactions.get(nid, [0, 0, 0])
            w.writerow([nid, n.x, n.y, n.z, *u, *r])
    L = model.member_lengths()
    with open(f"{prefix}_members.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "start", "end", "section", "length", "force_N", "stress_MPa"])
        for mid, m in sorted(model.members.items()):
            F = result.member_forces[mid]
            sig = F / _section_for(sections, m.section_id).area() if sections is not None else ""
            w.writerow([mid, m.start_node, m.end_node, m.section_id, L[mid], F, sig])
