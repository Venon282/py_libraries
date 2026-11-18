"""Tests for Edge.__eq__."""

from py_libraries.graph.Edge import Edge
from py_libraries.graph.Node import Node


def test_equal_edges_share_the_same_endpoints():
    assert Edge(Node('a'), Node('b')) == Edge(Node('a'), Node('b'))


def test_edges_with_different_endpoints_are_not_equal():
    assert Edge(Node('a'), Node('b')) != Edge(Node('a'), Node('c'))


def test_edge_direction_matters():
    assert Edge(Node('a'), Node('b')) != Edge(Node('b'), Node('a'))


def test_equality_ignores_weight_and_properties():
    a = Edge(Node('a'), Node('b'), weight=1, properties={'x': 1})
    b = Edge(Node('a'), Node('b'), weight=9, properties={'y': 2})

    assert a == b


def test_not_equal_to_a_non_edge():
    edge = Edge(Node('a'), Node('b'))

    assert (edge == 'a') is False
    assert (edge == 1) is False
