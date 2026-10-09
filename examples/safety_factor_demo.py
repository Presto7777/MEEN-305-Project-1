"""Phase 3 demo: safety factors for each layout / print orientation.
Usage: python examples/phase3_demo.py [diameter_mm]   (uniform round section)"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from trusskit import *
from trusskit import viz

d = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
sec, mat = SolidCircular(d), PLA_PLACEHOLDER
print(f"Uniform solid round section d = {d} mm, K = 1.0, required FS = 1.5\n")
print(f"{'layout':8}{'orient':>8}{'min FS':>8}  {'member':>6}  {'mode':<20}{'case':<14}{'#fail':>6}")
for layout in ("warren", "pratt", "howe"):
    model = generate_truss(LayoutParameters(layout=layout))
    sup = standard_supports(model)
    for orient in ("bottom", "top", "side"):
        frs = [analyze_failures(model, solve_truss(model, sup, create_load_case(model, c, sup),
                                                   sec, mat), sec, mat, orient, K=1.0)
               for c in "CD"]
        c = check_design(frs)
        print(f"{layout:8}{orient:>8}{c.min_fs:8.2f}  {c.member_id:>6}  {c.mode:<20}"
              f"{c.load_case:<14}{len(c.failing):>6}")

# detailed view of one case
model = generate_truss(LayoutParameters(layout="warren"))
sup = standard_supports(model)
frs = [analyze_failures(model, solve_truss(model, sup, create_load_case(model, c, sup), sec, mat),
                        sec, mat, "bottom") for c in "CD"]
print(); print(failure_summary(frs[0])); print(); print(failure_table(model, frs[0], top=10))
print(); print(check_design(frs))
env = envelope(frs)
fig = plt.figure(figsize=(12, 5))
viz.plot_safety_factors(model, env, 1.5, "3d", fig.add_subplot(1, 2, 1, projection="3d"),
                        "Worst FS over C and D")
viz.plot_safety_factors(model, env, 1.5, "front", fig.add_subplot(1, 2, 2))
fig.tight_layout(); fig.savefig("safety_factors.png", dpi=130)
