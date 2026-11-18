"""Tests for Node.isSource."""

from py_libraries.graph.Edge import Edge
from py_libraries.graph.Node import Node


def test_is_source_for_a_fresh_node():
    node = Node('a')

    assert node.isSource() is True


def test_is_not_source_once_it_has_an_incoming_edge():
    a = Node('a')
    b = Node('b')
    edge = Edge(a, b)
    a.addEdge(edge)
    b.addEdge(edge)

    assert b.isSource() is False
    assert a.isSource() is True
