"""Tests for the Edge.coordinate property."""

import pytest
from py_libraries.graph.Edge import Edge
from py_libraries.graph.Node import Node


def test_coordinate_trims_endpoints_to_node_borders():
    start = Node('a', x=0.0, y=0.0, radius=2.0)
    end = Node('b', x=10.0, y=0.0, radius=2.0)

    edge = Edge(start, end)

    assert edge.coordinate == ((2.0, 0.0), (8.0, 0.0))


def test_coordinate_on_a_diagonal():
    start = Node('a', x=0.0, y=0.0, radius=2.0)
    end = Node('b', x=3.0, y=4.0, radius=2.0)

    edge = Edge(start, end)

    (start_x, start_y), (end_x, end_y) = edge.coordinate
    assert start_x == pytest.approx(1.2)
    assert start_y == pytest.approx(1.6)
    assert end_x == pytest.approx(1.8)
    assert end_y == pytest.approx(2.4)


def test_coordinate_points_move_towards_each_other():
    start = Node('a', x=5.0, y=5.0, radius=1.0)
    end = Node('b', x=1.0, y=2.0, radius=1.0)

    edge = Edge(start, end)

    (start_x, start_y), (end_x, end_y) = edge.coordinate
    trimmed_start = (start_x - end.x) ** 2 + (start_y - end.y) ** 2
    raw_start = (5.0 - end.x) ** 2 + (5.0 - end.y) ** 2
    assert trimmed_start < raw_start
