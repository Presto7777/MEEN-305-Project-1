"""Preliminary comparison with ONE uniform placeholder section.
NOT a feasibility comparison: nothing here is sized to meet FS >= 1.5."""
from trusskit import *

sec = SolidCircular(3.0)
print(f"{'layout':8}{'nodes':>6}{'memb':>6}{'mass[g]':>9}"
      f"{'uz_C[mm]':>10}{'uz_D[mm]':>10}{'max T[N]':>10}{'max C[N]':>10}")
for layout in ("warren", "pratt", "howe", "triangular"):
    model = generate_truss(LayoutParameters(layout=layout))
    sup = standard_supports(model)
    out = {}
    for c in "CD":
        out[c] = solve_truss(model, sup, create_load_case(model, c, sup),
                             sec, PLA_PLACEHOLDER)
    forces = [f for r in out.values() for f in r.member_forces.values()]
    print(f"{layout:8}{len(model.nodes):6}{len(model.members):6}"
          f"{total_mass(model, sec, PLA_PLACEHOLDER):9.2f}"
          f"{out['C'].max_vertical_displacement():10.3f}"
          f"{out['D'].max_vertical_displacement():10.3f}"
          f"{max(forces):10.1f}{min(forces):10.1f}")