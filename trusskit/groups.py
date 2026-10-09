"""Member grouping: label members chord / web / brace so each group can have its own section."""
from dataclasses import replace
from .geometry import TrussModel

GROUPS = ("chord", "web", "brace")


def assign_groups(model, chord_cos=0.99, brace_z=0.01) -> TrussModel:
    """Return a NEW model whose members carry section_id in {"chord","web","brace"}.

    chord : runs along the span (|n_x| >= chord_cos)
    brace : horizontal but not a chord (transverse members, chord-plane diagonals)
    web   : everything else (verticals, web diagonals, slanted transverse bracing)
    """
    members = {}
    for mid, m in model.members.items():
        n = m.direction(model.nodes)
        if abs(n[0]) >= chord_cos:
            g = "chord"
        elif abs(n[2]) <= brace_z:
            g = "brace"
        else:
            g = "web"
        members[mid] = replace(m, section_id=g)
    return TrussModel(dict(model.nodes), members, model.name)

