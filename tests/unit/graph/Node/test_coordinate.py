"""Tests for the Node.coordinate property."""

from py_libraries.graph.Node import Node


def test_coordinate_returns_x_and_y():
    node = Node('a', x=1.5, y=-2.5)

    assert node.coordinate == (1.5, -2.5)


def test_coordinate_tracks_move():
    node = Node('a', x=1.0, y=1.0)

    node.move(1.0, 2.0)

    assert node.coordinate == (2.0, 3.0)
