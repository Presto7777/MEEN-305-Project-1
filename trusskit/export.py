"""Export a design (DesignSpec) to SolidWorks-style CSV files.

nodes.csv   : NodeID,X,Y,Z                        (N1.. ; coordinates in mm by default)
members.csv : MemberID,StartNode,EndNode,ProfilePath   (M1.. ; one weldment profile per member)

Format details copied from the reference files: CRLF line endings, no trailing newline,
integers written without a decimal point, 1-based IDs with N / M prefixes.
"""
import csv
import io
import ntpath
import os

from .design import DesignSpec
from .groups import GROUPS, assign_groups
from .layouts import LayoutParameters, generate_truss

DEFAULT_PROFILE = (r"C:\ProgramData\SOLIDWORKS\SOLIDWORKS 2025\weldment profiles"
                   r"\test\testa\test.SLDLFP")


def _num(v, decimals=6):
    s = f"{v:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def model_from_spec(spec: DesignSpec):
    """Rebuild the grouped TrussModel described by a spec (saved JSON holds no node list)."""
    return assign_groups(generate_truss(LayoutParameters(
        span=spec.span, height=spec.height, depth=spec.depth, panels=int(spec.panels),
        layout=spec.layout)))


def auto_profile_name(spec: DesignSpec, group: str, profile_dir: str) -> str:
    """e.g. <dir>\\hollow_rectangular_6x6x1.SLDLFP for the group's section type/params."""
    s = spec.sections[group]
    stem = s["kind"].replace(" ", "_") + "_" + "x".join(_num(p, 3) for p in s["params"])
    return ntpath.join(profile_dir, stem + ".SLDLFP")


def resolve_profiles(spec, default=DEFAULT_PROFILE, per_group=None, profile_dir=None) -> dict:
    """group -> profile path.  Priority: per_group override > auto name (profile_dir) > default."""
    per_group = per_group or {}
    out = {}
    for g in GROUPS:
        if per_group.get(g):
            out[g] = per_group[g]
        elif profile_dir:
            out[g] = auto_profile_name(spec, g, profile_dir)
        else:
            out[g] = default
    return out


def _to_text(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().rstrip("\r\n")          # reference files have no trailing newline


def solidworks_tables(spec, scale=1.0, shift_to_zero=True, offset=(0.0, 0.0, 0.0),
                      profiles=None, decimals=6):
    """Return (node_rows, member_rows, shift) as lists of lists.
    `shift` is the (dx, dy, dz) added to model coordinates (before scale) to reach the output."""
    model = model_from_spec(spec)
    profiles = profiles or resolve_profiles(spec)
    lo, _ = model.bounding_box()
    shift = [(-lo[i] if shift_to_zero else 0.0) + offset[i] for i in range(3)]
    node_rows = []
    for nid in sorted(model.nodes):
        n = model.nodes[nid]
        node_rows.append([f"N{nid + 1}"] + [_num((c + shift[i]) * scale, decimals)
                                            for i, c in enumerate((n.x, n.y, n.z))])
    member_rows = []
    for mid in sorted(model.members):
        m = model.members[mid]
        member_rows.append([f"M{mid + 1}", f"N{m.start_node + 1}", f"N{m.end_node + 1}",
                            profiles[m.section_id]])
    return node_rows, member_rows, shift


def write_solidworks_csv(spec, nodes_path, members_path, **kwargs):
    """Write the two CSV files; returns (node_rows, member_rows, shift)."""
    node_rows, member_rows, shift = solidworks_tables(spec, **kwargs)
    for path, header, rows in ((nodes_path, ["NodeID", "X", "Y", "Z"], node_rows),
                               (members_path, ["MemberID", "StartNode", "EndNode", "ProfilePath"],
                                member_rows)):
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        with open(path, "w", newline="") as fh:
            fh.write(_to_text(header, rows))
    return node_rows, member_rows, shift