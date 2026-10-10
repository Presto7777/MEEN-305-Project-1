"""Interactive explorer: sliders for width, height, panels and per-group cross-sections.
Usage: python examples/interactive_explorer.py [--layout warren|pratt|howe|triangular]
           [--orientation bottom|top|side] [--span 220] [--load designs/x.json]
           [--designs-dir designs]
Buttons: 'save design' (current design -> JSON), 'save trace' (all points -> JSON + CSV)."""
import argparse
from trusskit import DesignSpec
from trusskit.explorer import launch, LAYOUTS, ORIENTATIONS

ap = argparse.ArgumentParser()
ap.add_argument("--layout", default="warren", choices=LAYOUTS)
ap.add_argument("--orientation", default="bottom", choices=ORIENTATIONS)
ap.add_argument("--span", type=float, default=220.0)
ap.add_argument("--load", help="saved design JSON to start from")
ap.add_argument("--designs-dir", default="designs")
a = ap.parse_args()
spec = DesignSpec.load(a.load) if a.load else None
launch(layout=a.layout, orientation=a.orientation, span=a.span, spec=spec,
       designs_dir=a.designs_dir)