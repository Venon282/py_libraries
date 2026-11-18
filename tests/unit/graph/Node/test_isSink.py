"""Tests for Node.isSink."""

from py_libraries.graph.Edge import Edge
from py_libraries.graph.Node import Node


def test_is_sink_for_a_fresh_node():
    node = Node('a')

    assert node.isSink() is True


def test_is_not_sink_once_it_has_an_outgoing_edge():
    a = Node('a')
    b = Node('b')
    edge = Edge(a, b)
    a.addEdge(edge)
    b.addEdge(edge)

    assert a.isSink() is False
    assert b.isSink() is True
