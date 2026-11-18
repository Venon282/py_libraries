"""Tests for Node.removeEdge and Node.discardEdge."""

import pytest
from py_libraries.graph.Edge import Edge
from py_libraries.graph.Node import Node


def _linked_edge():
    start = Node('a')
    end = Node('b')
    edge = Edge(start, end)
    start.addEdge(edge)
    end.addEdge(edge)
    return start, end, edge


def test_removeEdge_unregisters_from_both_nodes():
    start, end, edge = _linked_edge()

    start.removeEdge(edge)
    end.removeEdge(edge)

    assert edge not in start.edges
    assert edge not in start.edges_out
    assert edge not in end.edges
    assert edge not in end.edges_in


def test_removeEdge_raises_keyerror_for_unknown_edge():
    node = Node('a')
    other = Edge(Node('x'), Node('y'))

    with pytest.raises(KeyError):
        node.removeEdge(other)


def test_discardEdge_is_silent_for_unknown_edge():
    node = Node('a')
    other = Edge(Node('x'), Node('y'))

    node.discardEdge(other)

    assert node.edges == set()


def test_discardEdge_unregisters_known_edge():
    start, end, edge = _linked_edge()

    start.discardEdge(edge)
    end.discardEdge(edge)

    assert edge not in start.edges
    assert edge not in end.edges
