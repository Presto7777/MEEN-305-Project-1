"""Material definitions.  Units throughout the package: mm, N, MPa (N/mm^2), g.

The PLA values below are GENERIC PLACEHOLDERS.  Replace them with values from
your filament data sheet or, better, from printed-coupon tests.
Strength allowables (tension, compression, interlayer shear) are deliberately
absent: they belong to the Phase 3 failure-analysis module.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    E: float        # elastic modulus [MPa = N/mm^2]
    density: float  # [g/mm^3]  (1.24 g/cm^3 = 1.24e-3 g/mm^3)

    def __post_init__(self):
        if self.E <= 0 or self.density <= 0:
            raise ValueError("E and density must be positive")


PLA_PLACEHOLDER = Material(name="PLA (generic placeholder)", E=3500.0, density=1.24e-3)
