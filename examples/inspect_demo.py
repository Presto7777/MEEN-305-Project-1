"""Inspect one candidate: prints summaries and writes PNG overviews + CSVs.
Usage: python examples/inspect_demo.py [warren|pratt|howe|triangular] [C|D]"""
import sys
import matplotlib
matplotlib.use("Agg")           # remove this line to get interactive windows
import matplotlib.pyplot as plt
from trusskit import *
from trusskit import viz

layout = sys.argv[1] if len(sys.argv) > 1 else "warren"
cond = sys.argv[2] if len(sys.argv) > 2 else "C"
sec = SolidCircular(3.0)

model = generate_truss(LayoutParameters(layout=layout))
sup = standard_supports(model)
lc = create_load_case(model, cond, sup)
res = solve_truss(model, sup, lc, sec, PLA_PLACEHOLDER)

print(model_summary(model, sec, PLA_PLACEHOLDER)); print()
print(result_summary(model, res)); print()
print(member_table(model, res, sec, top=12))

viz.inspect_model(model, res, sup, lc, path=f"{layout}_{cond}_overview.png")
fig = plt.figure(figsize=(7, 5)); viz.plot_deformed_3d(model, res, ax=fig.add_subplot(projection="3d"))
fig.savefig(f"{layout}_{cond}_deformed.png", dpi=130)
fig = plt.figure(figsize=(7, 5)); viz.plot_truss_3d(model, ax=fig.add_subplot(projection="3d"),
                                                   label_nodes=True, supports=sup, load_case=lc)
fig.savefig(f"{layout}_geometry_labeled.png", dpi=130)
export_csv(model, res, f"{layout}_{cond}", sec)
print("\nwrote PNGs and CSVs")