"""Cross-section classes (units: mm, mm^2, mm^4, mm^3)."""
from abc import ABC, abstractmethod
import math


class CrossSection(ABC):
    @abstractmethod
    def area(self) -> float: ...

    @abstractmethod
    def second_moments_of_area(self) -> tuple:
        """Return the two principal second moments (I_a, I_b)."""

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


class HollowCircular(CrossSection):
    def __init__(self, D: float, d: float):
        if not (0 <= d < D):
            raise ValueError("need 0 <= d < D")
        self.D, self.d = D, d

    def area(self):
        return math.pi / 4.0 * (self.D ** 2 - self.d ** 2)

    def second_moments_of_area(self):
        I = math.pi / 64.0 * (self.D ** 4 - self.d ** 4)
        return (I, I)


class Rectangular(CrossSection):
    def __init__(self, b: float, h: float):
        if b <= 0 or h <= 0:
            raise ValueError("b and h must be positive")
        self.b, self.h = b, h

    def area(self):
        return self.b * self.h

    def second_moments_of_area(self):
        return (self.b * self.h ** 3 / 12.0, self.h * self.b ** 3 / 12.0)
