"""Tests for the Edge.length property."""

import pytest
from py_libraries.graph.Edge import Edge
from py_libraries.graph.Node import Node


@pytest.mark.parametrize(
    ('start_xy', 'end_xy', 'expected'),
    [
        ((0.0, 0.0), (3.0, 4.0), 5.0),
        ((0.0, 0.0), (1.0, 0.0), 1.0),
        ((2.0, 2.0), (2.0, 2.0), 0.0),
    ],
)
def test_length_is_euclidean_distance(start_xy, end_xy, expected):
    start = Node('a', x=start_xy[0], y=start_xy[1])
    end = Node('b', x=end_xy[0], y=end_xy[1])

    edge = Edge(start, end)

    assert edge.length == pytest.approx(expected)
