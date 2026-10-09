# trusskit

A Python package for generating, solving and checking 3D pin-jointed truss
structures, built for a 220 mm FDM/PLA bridge project (conditions C and D,
required safety factor 1.5). Geometry, solver and failure checks are separate
modules: any truss, parametric or hand-entered, goes through the same analysis.

**Implemented:** geometry + validation, parametric layouts, supports/loads,
direct-stiffness solver, stress/buckling/print-orientation safety factors,
reports and plots. **Not yet implemented:** automatic member sizing, layout
sweeps/ranking, Excel report, self-weight, joint/load-introduction checks.

## Install and run

    pip install -e ".[dev]"          # numpy, pytest, matplotlib (use ".[plots]" for plots only)
    pytest                           # 79 tests
    python examples/run_demo.py      # layout comparison table (mass, deflection, forces)
    python examples/inspect_demo.py pratt D   # summaries, PNG overviews, CSVs
    python examples/phase3_demo.py 3          # safety factors for a 3 mm uniform section
python examples/interactive_explorer.py   # sliders: min FS vs. mass (matplotlib window)

Plotting is imported separately: `from trusskit import viz`.

## Conventions

| Item | Convention |
|---|---|
| Units | mm, N, MPa (N/mm²), g (density in g/mm³) |
| Axes | x along span, y across depth, z up |
| Axial force | tension **+**, compression **−** |
| Support reaction | force the support exerts on the structure (sums to −applied load) |
| Zero-force member | safety factor `inf`, mode `"n/a"` (no division by zero) |

## Quick start

```python
from trusskit import *

model = generate_truss(LayoutParameters(span=220, height=40, depth=30,
                                        panels=4, layout="warren"))
sup   = standard_supports(model)
sec   = SolidCircular(3.0)                      # or {"default": SolidCircular(3.0)}
mat   = PLA_PLACEHOLDER

frs = []
for cond in "CD":
    res = solve_truss(model, sup, create_load_case(model, cond, sup), sec, mat)
    frs.append(analyze_failures(model, res, sec, mat, orientation="bottom", K=1.0))
print(check_design(frs))                        # PASS/FAIL, min FS, member, mode, case
print(failure_table(model, frs[0], top=10))
```

## Package layout

| Module | Responsibility |
|---|---|
| `geometry.py` | `Node`, `Member`, `TrussModel`, validation |
| `sections.py` | `SolidCircular`, `HollowCircular`, `Rectangular`, `HollowRectangular`; `SECTION_TYPES` registry and `make_section` |
| `materials.py` | `Material`, `Strength`, PLA placeholders |
| `layouts.py` | `LayoutParameters`, `generate_truss` (Warren / Pratt / Howe / triangular, 3D) |
| `loads.py` | `Support`, `LoadCase`, `standard_supports`, `create_load_case` |
| `solver.py` | `solve_truss`, `AnalysisResult`, `total_mass`, solver exceptions |
| `failure.py` | stress, buckling, print-orientation checks, envelope, design check |
| `groups.py` | `assign_groups`: label members chord / web / brace |
| `explorer.py` | `evaluate_design` (pure logic) and the interactive `Explorer` window |
| `report.py` | text summaries, member tables, CSV export |
| `viz.py` | 3D/2D plots of geometry, forces, deformation, safety factors |

### Geometry (`geometry.py`)
- `Node(id, x, y, z)`, `Member(id, start_node, end_node, section_id="default")`.
  Members hold only a section *label*; geometry stores no forces or materials.
- `Member.length(nodes)`, `Member.direction(nodes)` (unit vector start→end).
- `TrussModel(nodes: dict, members: dict, name)`; `TrussModel.from_lists(...)`
  keeps duplicate IDs so `validate()` can report them.
- Queries: `member_lengths()`, `bounding_box()`, `node_index_map()` (ID → matrix
  index, IDs need not be contiguous), `find_nodes(x=, y=, z=, tol=)`,
  `connected_components()`.
- `validate(tol, span_limits, max_extent)` returns `(errors, warnings)`. Errors
  (block the solver): duplicate IDs, key/ID mismatch, missing endpoints,
  self-connected or zero-length members, duplicate connections (either
  direction), non-finite coordinates, disconnected components, build-volume
  overrun. Warnings: coincident nodes, span outside `span_limits`.
  Connectivity does not prove stability; the solver checks that.

### Layouts (`layouts.py`)
`generate_truss(LayoutParameters(span=220, height=40, depth=30, panels=4,
layout="warren"|"pratt"|"howe"|"triangular", section_id="default"))` builds two vertical side
trusses at y = ±depth/2 plus transverse members, alternating diagonal bracing in
the top and bottom chord planes, and transverse bracing in the two end bays
(prevents sideways racking). Output is validated before it is returned.
- **Warren:** top nodes staggered at panel mid-points (every bottom node gets a
  diagonal). Defaults give 18 nodes, 50 members.
- **Pratt / Howe:** aligned stations, verticals everywhere, diagonals leaning
  toward (Pratt) or away from (Howe) mid-span. Need an **even** panel count.
  Defaults give 20 nodes, 54 members.
- **Triangular:** one 3D truss with a triangular cross-section: two bottom chords
  at y = ±depth/2 and one apex chord at y = 0, z = height (`depth` = bottom width,
  `height` = apex height). A rigid triangle ring sits at every station and each of
  the three faces (bottom, left, right) has one alternating diagonal. It is exactly
  isostatic (members = 3·nodes − 6): 15 nodes, 39 members at defaults, any panel
  count. Because the bottom-face diagonal is single, the layout is not mirror-symmetric,
  so symmetric loads cause some lateral drift (~9% of vertical at defaults).
  Loads and supports use the two bottom-chord nodes at each station.
- Manual geometry uses the same `Node`/`Member`/`TrussModel` classes.

### Supports and loads (`loads.py`)
- `Support(node_id, (fix_x, fix_y, fix_z))`.
- `standard_supports(model)`: bottom-chord end nodes. Left: first node pinned,
  others fix x and z. Right: z only. Removes all six rigid-body modes. Replace
  with your real fixture.
- `create_load_case(model, "C"|"D", supports=None, mass_kg=8.0)`: C = one load at
  mid-span, D = two at quarter points, each m·g = 78.48 N downward on the bottom
  chord, split equally among nodes at the station. Positions are measured
  between support x-coordinates. Raises a clear error if no node exists there
  (use panels divisible by 4 for D).

### Solver (`solver.py`)
`solve_truss(model, supports, load_case, sections, material)` assembles the
dense 3N×3N stiffness matrix (`k = EA/L [[q,-q],[-q,q]]`, `q = nnᵀ`), partitions
free/fixed DOFs, solves `K_ff u_f = F_f` with `np.linalg.solve` (never an
inverse), then recovers reactions (`Ku − F`) and member forces (`EA/L · nᵀ(u_j−u_i)`).
`sections` is one `CrossSection` or a `{section_id: CrossSection}` dict.
Built-in checks:
- `UnstableStructureError`: λ_min/λ_max of K_ff below tolerance (mechanism or too few supports).
- `EquilibriumError`: ΣF or ΣM residual exceeds tolerance.
- Validation errors, missing support/load nodes, and non-finite results.

`AnalysisResult`: `displacements`, `reactions`, `member_forces`, `stiffness_ratio`,
`force_residual`, `moment_residual`, `max_vertical_displacement()`,
`max_lateral_displacement()`. `total_mass(model, sections, material)` returns
Σρ·A·L in grams (ideal members only; excludes joints, supports, slicer overhead).

### Materials and sections
- Sections give `area()`, `second_moments_of_area()`, `I_min()`, `volume(L)`,
  `describe()` and `min_feature()` (smallest wall/dimension, for printability warnings).
  `HollowRectangular(b, h, t)` has wall thickness `t` on all four sides (inner opening
  `(b-2t) x (h-2t)`, requires `2t <= min(b, h)`). `HollowCircular(D, d)` takes outer and
  inner diameters. `make_section(kind, params)` builds any type from the `SECTION_TYPES`
  registry (`"solid circular"`, `"hollow circular"` [D, t], `"rectangular"`,
  `"hollow rectangular"`).
- `Material(name, E, density, strength)`. `PLA_PLACEHOLDER`: E = 3500 MPa,
  1.24 g/cm³. `Strength` (generic PLA placeholders, replace with your coupon data):

| Property | Default |
|---|---|
| In-layer tension / compression | 50 / 60 MPa |
| Interlayer tension | 25 MPa |
| Interlayer shear | 20 MPa |
| `knockdown` (global factor on all of the above) | 1.0 |

### Failure analysis (`failure.py`)
`analyze_failures(model, result, sections, material, orientation="bottom", K=1.0,
fs_required=1.5)` returns a `FailureResult` of per-member `MemberFailure`
(force, stress, state, slenderness KL/r, layer cosine, σₙ, τ, `fs` dict,
governing FS and mode, flags). Checks per member:

| Mode | Applies to | Safety factor |
|---|---|---|
| tension / compression | tension / compression | S / \|σ\| (in-layer strength) |
| buckling | compression | P_cr / \|F\|: Euler if KL/r ≥ π√(2E/S_c), else Johnson parabola |
| interlayer tension | σₙ > 0 | S_zt / σₙ, σₙ = σ(n·l)² |
| interlayer shear | τ > 0 | S_zs / τ, τ = \|σ\|\|n·l\|√(1−(n·l)²) |

`l` is the layer-plane normal. Orientations: `"bottom"` and `"top"` (layers stack
in z), `"side"` (layers stack in y); also a `PrintOrientation` or a dict
`{member_id: orientation, "default": ...}`. Top and bottom share a normal; their
overhang/support differences are not modeled. Flags mark stocky members (Johnson
used) and KL/r > 150.

Combining cases: `envelope(list_of_FailureResult)` gives the worst FS per member
with its mode and load case; `check_design(...)` returns a `DesignCheck`
(min FS, member, mode, case, pass/fail, failing members). Also
`layer_tractions`, `critical_buckling_load`, `failure_table`, `failure_summary`.

### Reports and plots
- `report.py`: `model_summary`, `result_summary`, `member_table`,
  `export_csv(model, result, prefix, sections)` → `<prefix>_nodes.csv`
  (coordinates, displacements, reactions) and `<prefix>_members.csv`
  (force, stress).
- `viz.py`: `plot_truss_3d` (IDs, supports, loads), `plot_forces_3d` and
  `plot_view_2d` (red tension, blue compression, width ∝ |F|),
  `plot_deformed_3d` (auto-exaggerated), `plot_safety_factors` (colored by FS),
  `inspect_model` (4-panel overview). Pass `ax=` to compose your own figures.

## Interactive explorer (`explorer.py`)
`python examples/interactive_explorer.py [--layout ...] [--orientation ...] [--span 220]`
opens a matplotlib window. Sliders set width (y), height (z) and panels (snaps to
4, 8, 12, 16 so condition D has quarter-point load nodes). Cross-sections are set
**per member group**: pick a group (*edit group*: chord / web / brace), pick its
*section type* (solid circular, hollow circular, rectangular, hollow rectangular), and
the sliders below change to that type's parameters (e.g. width b, height h, wall t).
Each group remembers its parameters for every type. Radio buttons also select the
layout and print orientation.

Every change re-solves conditions C and D and plots one live point of **ideal member
mass vs. minimum safety factor** (log axis; green region = FS ≥ 1.5) with a trace of
past settings colored by layout ("clear trace" resets it). A front view colors each
member by its worst safety factor; a text panel lists each group's section and member
count, the governing member/mode/case and maximum deflections, and prints a
**PRINT WARNING** if any section feature (wall or dimension) is below 0.8 mm. Invalid
combinations (for example wall thicker than half the section) show a message and are
not added to the trace.

Groups come from `assign_groups(model)`: *chord* runs along the span, *brace* is
horizontal but not a chord (transverse members and chord-plane diagonals), *web* is
everything else. Headless use:

```python
secs = {"chord": make_section("hollow rectangular", [6, 6, 1.0]),
        "web":   make_section("hollow circular", [5, 0.8]),
        "brace": make_section("solid circular", [2])}
ev = evaluate_design("warren", span=220, height=40, depth=30, panels=4, sections=secs)
print(ev.mass, ev.min_fs, ev.check)
# or round bars only: evaluate_design(..., d_chord=5, d_web=4, d_brace=2)
```

Note: mass is ideal member volume; very thin hollow walls look light here but may
not print reliably, and local wall buckling is not modeled.

## Changing the design
- **Panels:** `LayoutParameters(panels=8)` (even for Pratt/Howe; multiple of 4 for condition D).
- **Per-member sections:** members are immutable; replace them with a new
  `section_id` (`dataclasses.replace(m, section_id="chord")`) and pass a
  `{section_id: CrossSection}` dict to the solver, failure check and mass.
  Nothing sizes sections automatically yet.

## Tests (`tests/`, 79 total)
- Geometry: 3-4-5 and 3D lengths, direction reversal, 4-panel/220 mm
  coordinates, all validation failures, bounding box.
- Solver: axial bar (δ = PL/EA), symmetric triangle vs. method of joints,
  equilibrium and reactions for all layouts and both conditions, linear
  scaling, mechanism and missing-support detection.
- Failure: Euler bar, in-layer tension, interlayer tension vs. orientation,
  45° tractions, Johnson regime, zero-force handling, FS ∝ 1/load, envelope,
  knockdown, thin-fails/thick-passes for every layout.
- Triangular layout: counts, isostatic check, geometry, reactions, lateral drift.
- Sections: `HollowRectangular` values and limits, registry, tube vs. solid at equal area.
- Explorer: member grouping, mixed section types, group/type controls, `evaluate_design` vs. the manual pipeline, FS/mass
  monotonic in diameters, GUI sliders/radios update the trace.
- Output: plots render and CSV/report functions run.

## Known limitations
- Ideal pin-jointed members: no bending, joint stiffness, or local effects at
  load points; no bracing between the end bays; self-weight and fixture mass omitted.
- Buckling uses K = 1.0 on the full member length (conservative for printed joints).
- Interlayer checks resolve uniaxial stress onto the layer plane; they ignore
  porosity, joints and stress concentrations. Strengths are generic placeholders:
  calibrate with coupons and the physical load test before trusting any FS.
- Linear-elastic, small-displacement analysis; no geometric nonlinearity.

## Next steps
Member sizing to FS ≥ 1.5, parameter sweep and ranking by mass/deflection,
Excel report, self-weight and fixture loads, joint and load-introduction checks,
and comparison with measured deflection from the printed bridge.