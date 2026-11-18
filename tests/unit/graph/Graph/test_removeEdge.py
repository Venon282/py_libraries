"""Tests for Graph.removeEdge and Graph.discardEdge."""

import pytest
from py_libraries.graph.Edge import Edge
from py_libraries.graph.Graph import Graph
from py_libraries.graph.Node import Node


def _graph_with_edge():
    graph = Graph()
    a = Node('a')
    b = Node('b')
    edge = Edge(a, b)
    graph.addEdge(edge)
    return graph, a, b, edge


def test_removeEdge_unregisters_from_graph_and_nodes():
    graph, a, b, edge = _graph_with_edge()

    graph.removeEdge(edge)

    assert edge not in graph.edges
    assert edge not in a.edges
    assert edge not in b.edges


def test_removeEdge_raises_keyerror_for_unknown_edge():
    graph, _, _, _ = _graph_with_edge()
    other = Edge(Node('x'), Node('y'))

    with pytest.raises(KeyError):
        graph.removeEdge(other)


def test_discardEdge_is_silent_for_unknown_edge():
    graph, _, _, _ = _graph_with_edge()
    other = Edge(Node('x'), Node('y'))

    graph.discardEdge(other)

    assert other not in graph.edges


def test_discardEdge_removes_a_known_edge():
    graph, a, b, edge = _graph_with_edge()

    graph.discardEdge(edge)

    assert edge not in graph.edges
    assert edge not in a.edges
    assert edge not in b.edges
