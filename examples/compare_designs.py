"""Compare several saved designs (JSON files): table + mass-vs-FS plot.
Usage: python examples/compare_designs.py designs/a.json designs/b.json [--out compare.png]
A trace file saved by the explorer ('save trace') can be passed with --trace."""
import argparse, json
import matplotlib
matplotlib.use("Agg")
from trusskit import DesignSpec
from trusskit.design_report import compare_designs

ap = argparse.ArgumentParser()
ap.add_argument("designs", nargs="*")
ap.add_argument("--trace", help="trace_*.json from the explorer (compares every point)")
ap.add_argument("--out", default="compare.png")
a = ap.parse_args()
specs, labels = [], []
for p in a.designs:
    s = DesignSpec.load(p); specs.append(s); labels.append(s.name or p.split("/")[-1][:-5])
if a.trace:
    for d in json.load(open(a.trace))["designs"]:
        specs.append(DesignSpec.from_dict(d)); labels.append(f"#{d['index']} {d['layout']}")
if not specs:
    ap.error("give design files and/or --trace")
table, _ = compare_designs(specs, labels, a.out)
print(table); print(f"\nplot saved to {a.out}")