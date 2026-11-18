"""Tests for the Node.nodesOut property."""

from py_libraries.graph.Edge import Edge
from py_libraries.graph.Node import Node


def test_nodes_out_is_empty_for_a_fresh_node():
    node = Node('a')

    assert node.nodesOut == []


def test_nodes_out_lists_successors():
    a = Node('a')
    b = Node('b')
    edge = Edge(a, b)
    a.addEdge(edge)
    b.addEdge(edge)

    assert a.nodesOut == [b]
    assert b.nodesOut == []
