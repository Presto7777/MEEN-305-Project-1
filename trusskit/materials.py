"""Material definitions.  Units throughout the package: mm, N, MPa (N/mm^2), g.

The PLA values below are GENERIC PLACEHOLDERS typical of data-sheet / literature
numbers for FDM PLA.  Replace them with values for your filament, printer and
settings -- ideally from printed-coupon tests in each print direction.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Strength:
    """Strength allowables [MPa].  'In-layer' = load path runs along the filament
    (member axis lying in the layer plane); 'interlayer' = across layer bonds."""
    tensile_inplane: float = 26.0
    compressive_inplane: float = 45.0
    interlayer_tension: float = 15.0   # ~50% of in-layer: typical FDM PLA knockdown
    interlayer_shear: float = 20.0
    knockdown: float = 1.0             # extra global factor (<1) for defects, joints, etc.

    def __post_init__(self):
        for v in (self.tensile_inplane, self.compressive_inplane,
                  self.interlayer_tension, self.interlayer_shear, self.knockdown):
            if v <= 0:
                raise ValueError("strength values must be positive")


PLA_STRENGTH = Strength()


@dataclass(frozen=True)
class Material:
    name: str
    E: float        # elastic modulus [MPa = N/mm^2]
    density: float  # [g/mm^3]  (1.24 g/cm^3 = 1.24e-3 g/mm^3)
    strength: Strength = PLA_STRENGTH

    def __post_init__(self):
        if self.E <= 0 or self.density <= 0:
            raise ValueError("E and density must be positive")


PLA_PLACEHOLDER = Material(name="PLA (generic placeholder)", E=3500.0, density=1.24e-3)
