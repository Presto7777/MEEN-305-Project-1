
"""Interactive explorer: sliders for width, height, panels and chord/web/brace diameters.
Usage: python examples/interactive_explorer.py [--layout warren|pratt|howe|triangular]
                                               [--orientation bottom|top|side] [--span 220]"""
import argparse
from trusskit.explorer import launch, LAYOUTS, ORIENTATIONS

ap = argparse.ArgumentParser()
ap.add_argument("--layout", default="warren", choices=LAYOUTS)
ap.add_argument("--orientation", default="bottom", choices=ORIENTATIONS)
ap.add_argument("--span", type=float, default=220.0)
a = ap.parse_args()
launch(layout=a.layout, orientation=a.orientation, span=a.span)


