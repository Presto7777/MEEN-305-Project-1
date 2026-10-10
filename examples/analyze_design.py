"""Re-run a saved design (JSON from the explorer) and write plots, CSVs and a summary.
Usage: python examples/analyze_design.py designs/warren_p4_20260101_120000.json [--out DIR]
                                         [--K 0.7 1.0 1.5] [--no-sensitivity] [--show]"""
import argparse, os
import matplotlib
ap = argparse.ArgumentParser()
ap.add_argument("design")
ap.add_argument("--out", default=None, help="output folder (default: <design>_analysis)")
ap.add_argument("--K", type=float, nargs="+", default=[0.7, 1.0, 1.5])
ap.add_argument("--no-sensitivity", action="store_true")
ap.add_argument("--show", action="store_true", help="open the safety-factor figure afterwards")
a = ap.parse_args()
matplotlib.use("TkAgg" if a.show else "Agg")
import matplotlib.pyplot as plt
from trusskit import DesignSpec
from trusskit.design_report import analyze_design

out = a.out or os.path.splitext(a.design)[0] + "_analysis"
print(analyze_design(DesignSpec.load(a.design), out, K_values=a.K, sensitivity=not a.no_sensitivity))
print(f"\nfiles written to {out}/")
if a.show:
    import matplotlib.image as mpimg
    plt.imshow(mpimg.imread(os.path.join(out, "safety_factors.png"))); plt.axis("off"); plt.show()