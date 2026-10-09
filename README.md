# trusskit (Phases 1-2)

Units: mm, N, MPa, g. Axes: x span, y depth, z up. Tension positive.

    pip install -e ".[dev]"
    pytest
    python examples/run_demo.py

Modules: geometry (Node/Member/TrussModel/validate), sections, materials
(PLA values are placeholders), layouts (warren/pratt/howe 3D generator),
loads (supports, conditions C/D), solver (direct stiffness, stability and
equilibrium checks).

Known simplifications: end-bay transverse bracing only; loads on bottom chord
split across both side trusses; self-weight omitted; no strength/buckling yet (Phase 3).
