"""Convert a saved design (JSON from the explorer) into SolidWorks node/member CSV files.

    python examples/export_solidworks.py designs/warren_p4_xxx.json
    python examples/export_solidworks.py design.json --nodes nodes.csv --members members2.csv
    python examples/export_solidworks.py design.json --profile-chord "C:\\...\\chord.SLDLFP" \\
        --profile-web "C:\\...\\web.SLDLFP" --profile-brace "C:\\...\\brace.SLDLFP"
    python examples/export_solidworks.py design.json --profile-dir "C:\\...\\weldment profiles\\mine"
    python examples/export_solidworks.py designs/trace_xxx.json --index 3     # a point from a trace

The saved JSON stores design parameters (not node lists), so the geometry is rebuilt with
trusskit and written as:
    nodes.csv   : NodeID,X,Y,Z
    members.csv : MemberID,StartNode,EndNode,ProfilePath
"""
import argparse
import json
import os
import sys

from trusskit import DesignSpec
from trusskit.export import write_solidworks_csv, resolve_profiles, DEFAULT_PROFILE, GROUPS

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument("design", help="design JSON (single design, or a trace file with --index)")
ap.add_argument("--index", type=int, default=None,
                help="for trace files: which design to export (default: last)")
ap.add_argument("--nodes", help="nodes CSV path (default: <json stem>_nodes.csv)")
ap.add_argument("--members", help="members CSV path (default: <json stem>_members.csv)")
ap.add_argument("--out-dir", help="folder for the default file names")
ap.add_argument("--scale", type=float, default=1.0,
                help="multiply coordinates (model is in mm; 1 = mm, 0.001 = m, 0.0393701 = in)")
ap.add_argument("--offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("DX", "DY", "DZ"),
                help="translation added before scaling, in model mm")
ap.add_argument("--keep-origin", action="store_true",
                help="do not shift the model so its minimum x, y, z is 0 "
                     "(by default y is shifted by depth/2 so no coordinate is negative)")
ap.add_argument("--decimals", type=int, default=6)
ap.add_argument("--profile", default=DEFAULT_PROFILE,
                help="profile path used for every member unless overridden")
for g in GROUPS:
    ap.add_argument(f"--profile-{g}", help=f"profile path for {g} members")
ap.add_argument("--profile-dir",
                help="folder: use auto-named profiles like <dir>\\hollow_rectangular_6x6x1.SLDLFP "
                     "(for groups without an explicit --profile-<group>)")
a = ap.parse_args()

with open(a.design) as fh:
    data = json.load(fh)
if "designs" in data:                                   # trace file
    idx = len(data["designs"]) - 1 if a.index is None else a.index
    try:
        spec = DesignSpec.from_dict(data["designs"][idx])
    except IndexError:
        sys.exit(f"--index {idx} out of range (trace has {len(data['designs'])} designs)")
    tag = f"design {idx} of trace"
else:
    spec = DesignSpec.from_dict(data)
    tag = spec.name or "design"

stem = os.path.splitext(os.path.basename(a.design))[0] + (f"_{a.index}" if "designs" in data and a.index is not None else "")
base = os.path.join(a.out_dir or os.path.dirname(os.path.abspath(a.design)), stem)
nodes_path = a.nodes or base + "_nodes.csv"
members_path = a.members or base + "_members.csv"

profiles = resolve_profiles(spec, a.profile, {g: getattr(a, f"profile_{g}") for g in GROUPS},
                            a.profile_dir)
node_rows, member_rows, shift = write_solidworks_csv(
    spec, nodes_path, members_path, scale=a.scale, shift_to_zero=not a.keep_origin,
    offset=tuple(a.offset), profiles=profiles, decimals=a.decimals)

print(f"Exported {tag}: {spec.layout}, span {spec.span:g}, height {spec.height:g}, "
      f"width {spec.depth:g}, panels {spec.panels}")
print(f"  {len(node_rows)} nodes   -> {nodes_path}")
print(f"  {len(member_rows)} members -> {members_path}")
print(f"  coordinates: model mm x {a.scale:g}, shifted by "
      f"({shift[0]:g}, {shift[1]:g}, {shift[2]:g}) mm before scaling")
print("\nProfiles used (create/point these in SolidWorks):")
from trusskit.export import model_from_spec
model = model_from_spec(spec)
for g in GROUPS:
    n = sum(1 for m in model.members.values() if m.section_id == g)
    s = spec.build_sections()[g]
    print(f"  {g:<5} x{n:<3} {s.describe():<32} {profiles[g]}")
