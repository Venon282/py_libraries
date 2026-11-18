"""Tests for the Node.nodesIn property."""

from py_libraries.graph.Edge import Edge
from py_libraries.graph.Node import Node


def test_nodes_in_is_empty_for_a_fresh_node():
    node = Node('a')

    assert node.nodesIn == []


def test_nodes_in_lists_predecessors():
    a = Node('a')
    b = Node('b')
    edge = Edge(a, b)
    a.addEdge(edge)
    b.addEdge(edge)

    assert b.nodesIn == [a]
    assert a.nodesIn == []
