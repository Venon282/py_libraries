"""Tests for Node.move."""

from py_libraries.graph.Node import Node


def test_move_shifts_both_coordinates():
    node = Node('a', x=1.0, y=2.0)

    node.move(3.0, -4.0)

    assert node.x == 4.0
    assert node.y == -2.0


def test_move_with_zero_offset_keeps_position():
    node = Node('a', x=5.0, y=6.0)

    node.move(0.0, 0.0)

    assert node.x == 5.0
    assert node.y == 6.0
