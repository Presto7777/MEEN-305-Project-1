import math
import pytest
from trusskit import *


def test_hollow_rectangular_values():
    s = HollowRectangular(10, 20, 2)            # inner 6 x 16
    assert s.area() == pytest.approx(200 - 96)
    Ia, Ib = s.second_moments_of_area()
    assert Ia == pytest.approx((10 * 20 ** 3 - 6 * 16 ** 3) / 12)
    assert Ib == pytest.approx((20 * 10 ** 3 - 16 * 6 ** 3) / 12)
    assert s.I_min() == pytest.approx(Ib)
    assert s.min_feature() == 2


def test_hollow_rectangular_limit_is_solid_rectangle():
    s, r = HollowRectangular(10, 20, 5), Rectangular(10, 20)    # 2t == min(b, h)
    assert s.area() == pytest.approx(r.area())
    assert s.second_moments_of_area() == pytest.approx(r.second_moments_of_area())


@pytest.mark.parametrize("args", [(10, 20, 0), (10, 20, -1), (0, 5, 1), (10, 20, 5.1)])
def test_hollow_rectangular_invalid(args):
    with pytest.raises(ValueError):
        HollowRectangular(*args)


def test_tube_beats_solid_at_equal_area():
    tube = make_section("hollow circular", [10.0, 1.0])         # D=10, t=1
    d = math.sqrt(4 * tube.area() / math.pi)
    assert SolidCircular(d).area() == pytest.approx(tube.area())
    assert tube.I_min() > SolidCircular(d).I_min()


@pytest.mark.parametrize("kind,params,desc", [
    ("solid circular", [4.0], "solid round"),
    ("hollow circular", [8.0, 1.0], "tube"),
    ("rectangular", [4.0, 6.0], "rect 4.0x6.0"),
    ("hollow rectangular", [6.0, 8.0, 1.0], "rect tube"),
])
def test_make_section_all_kinds(kind, params, desc):
    s = make_section(kind, params)
    assert s.area() > 0 and s.I_min() > 0 and desc in s.describe()
    assert len(SECTION_TYPES[kind].params) == len(params)
    assert len(SECTION_TYPES[kind].defaults(3.0)) == len(params)


def test_make_section_errors():
    with pytest.raises(ValueError):
        make_section("hexagon", [1.0])
    with pytest.raises(ValueError):
        make_section("rectangular", [1.0])
    with pytest.raises(ValueError):
        make_section("hollow circular", [4.0, 2.5])             # wall > D/2