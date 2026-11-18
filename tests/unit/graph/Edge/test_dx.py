"""Tests for the Edge.dx and Edge.dy properties."""

import pytest
from py_libraries.graph.Edge import Edge
from py_libraries.graph.Node import Node


@pytest.mark.parametrize(
    ('start_xy', 'end_xy', 'expected'),
    [
        ((0.0, 0.0), (10.0, 4.0), (10.0, 4.0)),
        ((5.0, 5.0), (1.0, 2.0), (-4.0, -3.0)),
        ((3.0, 3.0), (3.0, 3.0), (0.0, 0.0)),
    ],
)
def test_dx_and_dy_are_end_minus_start(start_xy, end_xy, expected):
    start = Node('a', x=start_xy[0], y=start_xy[1])
    end = Node('b', x=end_xy[0], y=end_xy[1])

    edge = Edge(start, end)

    assert edge.dx == expected[0]
    assert edge.dy == expected[1]
