"""Cross-section classes (units: mm, mm^2, mm^4) and a registry used by the explorer.

Every section provides area(), second_moments_of_area() (two principal values),
I_min(), volume(L), describe() and min_feature() (smallest wall / dimension, mm,
useful for FDM printability checks).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
import math


class CrossSection(ABC):
    @abstractmethod
    def area(self) -> float: ...

    @abstractmethod
    def second_moments_of_area(self) -> tuple:
        """Return the two principal second moments (I_a, I_b)."""

    @abstractmethod
    def describe(self) -> str: ...

    @abstractmethod
    def min_feature(self) -> float:
        """Smallest wall thickness or solid dimension [mm]."""

    def I_min(self) -> float:
        return min(self.second_moments_of_area())

    def volume(self, length: float) -> float:
        return self.area() * length


class SolidCircular(CrossSection):
    def __init__(self, d: float):
        if d <= 0:
            raise ValueError("diameter must be positive")
        self.d = d

    def area(self):
        return math.pi * self.d ** 2 / 4.0

    def second_moments_of_area(self):
        I = math.pi * self.d ** 4 / 64.0
        return (I, I)

    def describe(self):
        return f"solid round Ø{self.d:.1f}"

    def min_feature(self):
        return self.d


class HollowCircular(CrossSection):
    """Tube with outer diameter D and inner diameter d (wall t = (D - d)/2)."""

    def __init__(self, D: float, d: float):
        if not (0 <= d < D):
            raise ValueError("need 0 <= d < D")
        self.D, self.d = D, d

    def area(self):
        return math.pi / 4.0 * (self.D ** 2 - self.d ** 2)

    def second_moments_of_area(self):
        I = math.pi / 64.0 * (self.D ** 4 - self.d ** 4)
        return (I, I)

    def describe(self):
        return f"tube Ø{self.D:.1f}, wall {(self.D - self.d) / 2:.2f}"

    def min_feature(self):
        return (self.D - self.d) / 2.0 if self.d > 0 else self.D


class Rectangular(CrossSection):
    def __init__(self, b: float, h: float):
        if b <= 0 or h <= 0:
            raise ValueError("b and h must be positive")
        self.b, self.h = b, h

    def area(self):
        return self.b * self.h

    def second_moments_of_area(self):
        return (self.b * self.h ** 3 / 12.0, self.h * self.b ** 3 / 12.0)

    def describe(self):
        return f"rect {self.b:.1f}x{self.h:.1f}"

    def min_feature(self):
        return min(self.b, self.h)


class HollowRectangular(CrossSection):
    """Rectangular tube: outer b (width) x h (height), wall thickness t on ALL four sides.
    Inner opening is (b - 2t) x (h - 2t); needs 2t <= min(b, h)."""

    def __init__(self, b: float, h: float, t: float):
        if b <= 0 or h <= 0 or t <= 0:
            raise ValueError("b, h and t must be positive")
        if 2 * t > min(b, h):
            raise ValueError(f"wall too thick: need 2t <= min(b, h) (2t = {2 * t:g}, "
                             f"min(b,h) = {min(b, h):g})")
        self.b, self.h, self.t = b, h, t

    def area(self):
        return self.b * self.h - (self.b - 2 * self.t) * (self.h - 2 * self.t)

    def second_moments_of_area(self):
        b, h, t = self.b, self.h, self.t
        bi, hi = b - 2 * t, h - 2 * t
        return ((b * h ** 3 - bi * hi ** 3) / 12.0,    # bending about the b-parallel axis
                (h * b ** 3 - hi * bi ** 3) / 12.0)

    def describe(self):
        return f"rect tube {self.b:.1f}x{self.h:.1f}, wall {self.t:.2f}"

    def min_feature(self):
        return self.t


# ----------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class SectionType:
    key: str
    params: tuple            # ((label, min, max, step), ...) one entry per parameter
    build: Callable          # list[float] -> CrossSection
    defaults: Callable       # nominal size s [mm] -> list[float]


_DIM = (0.5, 15.0, 0.1)
_THK = (0.2, 5.0, 0.1)


def _tube(p):
    D, t = p
    if 2 * t > D:
        raise ValueError(f"wall too thick: need 2t <= D (2t = {2 * t:g}, D = {D:g})")
    return HollowCircular(D, D - 2 * t)


SECTION_TYPES = {
    "solid circular": SectionType(
        "solid circular", (("Ø d",) + _DIM,),
        lambda p: SolidCircular(p[0]), lambda s: [s]),
    "hollow circular": SectionType(
        "hollow circular", (("outer Ø D",) + _DIM, ("wall t",) + _THK),
        _tube, lambda s: [1.5 * s, 0.25 * s]),
    "rectangular": SectionType(
        "rectangular", (("width b",) + _DIM, ("height h",) + _DIM),
        lambda p: Rectangular(p[0], p[1]), lambda s: [s, s]),
    "hollow rectangular": SectionType(
        "hollow rectangular", (("width b",) + _DIM, ("height h",) + _DIM, ("wall t",) + _THK),
        lambda p: HollowRectangular(p[0], p[1], p[2]), lambda s: [1.5 * s, 1.5 * s, 0.25 * s]),
}


def make_section(kind: str, params) -> CrossSection:
    """Build a section from a registry key and its parameter list."""
    if kind not in SECTION_TYPES:
        raise ValueError(f"unknown section type '{kind}'; choose from {list(SECTION_TYPES)}")
    st = SECTION_TYPES[kind]
    if len(params) != len(st.params):
        raise ValueError(f"'{kind}' needs {len(st.params)} parameter(s), got {len(params)}")
    return st.build([float(x) for x in params])