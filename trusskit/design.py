"""Design specification (save/load as JSON) and the evaluation pipeline.

A DesignSpec is everything needed to rebuild and re-analyze a design: layout, geometry,
per-group cross-sections, print orientation, K, loads and material.  Specs are what the
interactive explorer saves, and what analysis scripts reload.
"""
from dataclasses import dataclass, field, asdict
import datetime
import json
import os

from .layouts import LayoutParameters, generate_truss
from .groups import assign_groups, GROUPS
from .sections import SolidCircular, SECTION_TYPES, make_section
from .materials import Material, Strength, PLA_PLACEHOLDER
from .loads import standard_supports, create_load_case
from .solver import solve_truss, total_mass
from .failure import analyze_failures, envelope, check_design

LAYOUTS = ("warren", "pratt", "howe", "triangular")
ORIENTATIONS = ("bottom", "top", "side")
GROUP_BASE_SIZE = {"chord": 4.0, "web": 3.0, "brace": 2.0}   # nominal size for default sections
SCHEMA = 1


# ----------------------------------------------------------------------------- evaluation
@dataclass
class DesignEvaluation:
    layout: str
    orientation: str
    panels: int
    section_desc: dict       # group -> text description of its section
    min_features: dict       # group -> smallest wall/dimension [mm]
    group_counts: dict
    model: object
    sections: dict           # group -> CrossSection
    mass: float              # g (ideal members)
    check: object            # DesignCheck (min FS over C and D)
    envelope: dict           # member -> EnvelopeEntry
    max_uz: dict             # {"C": mm, "D": mm}
    max_uy: dict
    fs_required: float
    results: dict = field(default_factory=dict)    # "C"/"D" -> AnalysisResult
    failures: dict = field(default_factory=dict)   # "C"/"D" -> FailureResult

    @property
    def min_fs(self):
        return self.check.min_fs

    def summary(self) -> dict:
        """Plain-data summary (JSON-safe)."""
        c = self.check
        return {"mass_g": float(self.mass), "min_fs": float(c.min_fs), "passes": bool(c.passes),
                "governing_member": int(c.member_id), "governing_mode": c.mode,
                "governing_case": c.load_case, "n_members_below_required": len(c.failing),
                "max_uz_mm": {k: float(v) for k, v in self.max_uz.items()},
                "max_uy_mm": {k: float(v) for k, v in self.max_uy.items()},
                "n_nodes": len(self.model.nodes), "n_members": len(self.model.members),
                "group_counts": dict(self.group_counts), "section_desc": dict(self.section_desc)}


def evaluate_design(layout, span, height, depth, panels, d_chord=None, d_web=None, d_brace=None,
                    orientation="bottom", K=1.0, material=PLA_PLACEHOLDER,
                    fs_required=1.5, mass_kg=8.0, sections=None) -> DesignEvaluation:
    """Generate, size, solve C and D, and check safety factors.

    Sizing: either give round diameters d_chord/d_web/d_brace, or pass
    sections={"chord": CrossSection, "web": ..., "brace": ...} (takes precedence).
    """
    model = assign_groups(generate_truss(LayoutParameters(
        span=span, height=height, depth=depth, panels=int(panels), layout=layout)))
    if sections is None:
        if None in (d_chord, d_web, d_brace):
            raise ValueError("give d_chord, d_web, d_brace or a sections dict")
        sections = {"chord": SolidCircular(d_chord), "web": SolidCircular(d_web),
                    "brace": SolidCircular(d_brace)}
    missing = [g for g in GROUPS if g not in sections]
    if missing:
        raise ValueError(f"sections missing for groups {missing}")
    sup = standard_supports(model)
    frs, uz, uy, results, failures = [], {}, {}, {}, {}
    for cond in "CD":
        res = solve_truss(model, sup, create_load_case(model, cond, sup, mass_kg),
                          sections, material)
        fr = analyze_failures(model, res, sections, material, orientation, K, fs_required)
        frs.append(fr)
        results[cond], failures[cond] = res, fr
        uz[cond], uy[cond] = res.max_vertical_displacement(), res.max_lateral_displacement()
    counts = {g: sum(1 for m in model.members.values() if m.section_id == g) for g in GROUPS}
    return DesignEvaluation(
        layout, orientation, int(panels),
        {g: sections[g].describe() for g in GROUPS},
        {g: sections[g].min_feature() for g in GROUPS},
        counts, model, sections, total_mass(model, sections, material),
        check_design(frs, fs_required), envelope(frs), uz, uy, fs_required,
        results, failures)


# ----------------------------------------------------------------------------- spec
def _default_sections():
    return {g: {"kind": "solid circular", "params": [GROUP_BASE_SIZE[g]]} for g in GROUPS}


def material_to_dict(m: Material) -> dict:
    return asdict(m)


def material_from_dict(d: dict) -> Material:
    return Material(d["name"], float(d["E"]), float(d["density"]),
                    Strength(**d.get("strength", {})))


@dataclass
class DesignSpec:
    """Serializable design.  sections: {group: {"kind": <SECTION_TYPES key>, "params": [..]}}."""
    layout: str = "warren"
    span: float = 220.0
    height: float = 40.0
    depth: float = 30.0
    panels: int = 4
    sections: dict = field(default_factory=_default_sections)
    orientation: str = "bottom"
    K: float = 1.0
    fs_required: float = 1.5
    mass_kg: float = 8.0
    material: Material = PLA_PLACEHOLDER
    name: str = ""
    notes: str = ""

    def __post_init__(self):
        for g in GROUPS:
            if g not in self.sections:
                raise ValueError(f"sections missing group '{g}'")
            kind = self.sections[g]["kind"]
            if kind not in SECTION_TYPES:
                raise ValueError(f"unknown section type '{kind}' for {g}")
            if len(self.sections[g]["params"]) != len(SECTION_TYPES[kind].params):
                raise ValueError(f"wrong number of parameters for {g} ({kind})")

    # ---- building / analysis
    def build_sections(self) -> dict:
        return {g: make_section(s["kind"], s["params"]) for g, s in self.sections.items()}

    def evaluate(self) -> DesignEvaluation:
        return evaluate_design(self.layout, self.span, self.height, self.depth, self.panels,
                               orientation=self.orientation, K=self.K, material=self.material,
                               fs_required=self.fs_required, mass_kg=self.mass_kg,
                               sections=self.build_sections())

    # ---- (de)serialization
    def to_dict(self, evaluation: DesignEvaluation = None) -> dict:
        d = {"schema": SCHEMA, "name": self.name, "notes": self.notes,
             "layout": self.layout, "span": self.span, "height": self.height,
             "depth": self.depth, "panels": int(self.panels),
             "orientation": self.orientation, "K": self.K, "fs_required": self.fs_required,
             "mass_kg": self.mass_kg,
             "sections": {g: {"kind": s["kind"], "params": [float(x) for x in s["params"]]}
                          for g, s in self.sections.items()},
             "material": material_to_dict(self.material)}
        if evaluation is not None:
            d["results"] = evaluation.summary()      # informational only; ignored on load
        return d

    @classmethod
    def from_dict(cls, d: dict):
        if d.get("schema", SCHEMA) > SCHEMA:
            raise ValueError(f"design file schema {d['schema']} is newer than this code ({SCHEMA})")
        kw = {k: d[k] for k in ("layout", "span", "height", "depth", "panels", "orientation",
                                "K", "fs_required", "mass_kg", "name", "notes") if k in d}
        if "sections" in d:
            kw["sections"] = {g: {"kind": s["kind"], "params": [float(x) for x in s["params"]]}
                              for g, s in d["sections"].items()}
        if "material" in d:
            kw["material"] = material_from_dict(d["material"])
        if "panels" in kw:
            kw["panels"] = int(kw["panels"])
        return cls(**kw)

    def save(self, path, evaluation: DesignEvaluation = None):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(evaluation), fh, indent=2)
        return path

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


def timestamped_path(directory, stem, ext=".json"):
    """directory/stem_YYYYmmdd_HHMMSS[.n].ext that does not exist yet."""
    os.makedirs(directory, exist_ok=True)
    base = os.path.join(directory, f"{stem}_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    path, n = base + ext, 1
    while os.path.exists(path):
        path, n = f"{base}_{n}{ext}", n + 1
    return path