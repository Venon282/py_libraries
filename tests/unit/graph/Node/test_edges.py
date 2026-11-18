"""Tests for the Node.edges, Node.edges_in and Node.edges_out properties."""

import pytest
from py_libraries.graph.Edge import Edge
from py_libraries.graph.Node import Node


def test_edges_is_empty_for_a_fresh_node():
    node = Node('a')

    assert node.edges == set()
    assert node.edges_in == set()
    assert node.edges_out == set()


def test_edges_reflects_registered_edges():
    start = Node('a')
    end = Node('b')
    edge = Edge(start, end)
    start.addEdge(edge)
    end.addEdge(edge)

    assert start.edges == {edge}
    assert start.edges_out == {edge}
    assert start.edges_in == set()
    assert end.edges == {edge}
    assert end.edges_in == {edge}
    assert end.edges_out == set()


@pytest.mark.parametrize('attr', ['edges', 'edges_in', 'edges_out'])
def test_edge_properties_return_live_sets(attr):
    start = Node('a')
    end = Node('b')
    edge = Edge(start, end)

    result = getattr(start, attr)
    result.add(edge)

    assert edge in getattr(start, attr)
