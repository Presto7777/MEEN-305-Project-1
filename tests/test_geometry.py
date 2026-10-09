import numpy as np
import pytest
from trusskit import Node, Member, TrussModel, LayoutParameters, generate_truss


def test_345_member():
    nodes = {0: Node(0, 0, 0, 0), 1: Node(1, 3, 4, 0)}
    m = Member(0, 0, 1)
    assert m.length(nodes) == pytest.approx(5.0)
    assert np.allclose(m.direction(nodes), [0.6, 0.8, 0.0])


def test_3d_member():
    nodes = {0: Node(0, 1, 2, 3), 1: Node(1, 3, 6, 7)}   # d = (2,4,4), L = 6
    m = Member(0, 0, 1)
    assert m.length(nodes) == pytest.approx(6.0)
    assert np.allclose(m.direction(nodes), [1 / 3, 2 / 3, 2 / 3])


def test_reversed_endpoints():
    nodes = {0: Node(0, 0, 0, 0), 1: Node(1, 3, 4, 0)}
    a, b = Member(0, 0, 1), Member(1, 1, 0)
    assert a.length(nodes) == pytest.approx(b.length(nodes))
    assert np.allclose(a.direction(nodes), -b.direction(nodes))


def test_four_panel_220():
    m = generate_truss(LayoutParameters(span=220, panels=4, layout="pratt"))
    xs = sorted({round(n.x, 9) for n in m.nodes.values()})
    assert xs == [0, 55, 110, 165, 220]
    errors, _ = m.validate()
    assert errors == []


def test_warren_top_nodes_staggered():
    m = generate_truss(LayoutParameters(span=220, panels=4, layout="warren"))
    top = sorted({n.x for n in m.nodes.values() if n.z > 0})
    assert top == pytest.approx([27.5, 82.5, 137.5, 192.5])


def test_duplicate_ids_and_members():
    nodes = [Node(0, 0, 0, 0), Node(0, 1, 0, 0), Node(2, 2, 0, 0)]
    members = [Member(0, 0, 2), Member(1, 2, 0)]
    m = TrussModel.from_lists(nodes, members)
    errors, _ = m.validate()
    assert any("duplicate node id" in e for e in errors)
    assert any("duplicates member" in e for e in errors)


def test_invalid_reference_and_zero_length():
    nodes = {0: Node(0, 0, 0, 0), 1: Node(1, 0, 0, 0), 2: Node(2, 1, 0, 0)}
    members = {0: Member(0, 0, 9), 1: Member(1, 0, 1), 2: Member(2, 2, 2)}
    errors, _ = TrussModel(nodes, members).validate()
    assert any("missing node 9" in e for e in errors)
    assert any("zero length" in e for e in errors)
    assert any("itself" in e for e in errors)


def test_disconnected():
    nodes = {i: Node(i, i, 0, 0) for i in range(4)}
    members = {0: Member(0, 0, 1), 1: Member(1, 2, 3)}
    errors, _ = TrussModel(nodes, members).validate()
    assert any("disconnected" in e for e in errors)


def test_nan_coordinates():
    nodes = {0: Node(0, 0, 0, 0), 1: Node(1, float("nan"), 0, 0)}
    errors, _ = TrussModel(nodes, {0: Member(0, 0, 1)}).validate()
    assert any("non-finite" in e for e in errors)


def test_bounding_box():
    m = generate_truss(LayoutParameters(span=220, height=40, depth=30))
    lo, hi = m.bounding_box()
    assert np.allclose(lo, [0, -15, 0]) and np.allclose(hi, [220, 15, 40])


def test_odd_panels_pratt_rejected():
    with pytest.raises(ValueError):
        generate_truss(LayoutParameters(panels=3, layout="pratt"))
