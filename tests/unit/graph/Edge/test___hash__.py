"""Tests for Edge.__hash__."""

from py_libraries.graph.Edge import Edge
from py_libraries.graph.Node import Node


def test_hash_equals_hash_of_endpoint_pair():
    start = Node('a')
    end = Node('b')

    edge = Edge(start, end)

    assert hash(edge) == hash((start, end))


def test_edges_with_same_endpoints_hash_equal():
    assert hash(Edge(Node('a'), Node('b'))) == hash(Edge(Node('a'), Node('b')))
