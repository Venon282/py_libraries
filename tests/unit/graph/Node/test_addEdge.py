"""Tests for Node.addEdge."""

import pytest
from py_libraries.graph.Edge import Edge
from py_libraries.graph.Node import Node


def test_addEdge_registers_start_and_end():
    start = Node('a')
    end = Node('b')
    edge = Edge(start, end)

    start.addEdge(edge)
    end.addEdge(edge)

    assert edge in start.edges
    assert edge in start.edges_out
    assert edge not in start.edges_in
    assert edge in end.edges
    assert edge in end.edges_in
    assert edge not in end.edges_out


def test_addEdge_registers_self_loop_on_both_sides():
    node = Node('a')
    edge = Edge(node, node)

    node.addEdge(edge)

    assert edge in node.edges
    assert edge in node.edges_in
    assert edge in node.edges_out


def test_addEdge_ignores_unrelated_node():
    node = Node('a')
    other = Edge(Node('x'), Node('y'))

    node.addEdge(other)

    assert other not in node.edges
    assert node.edges == set()


@pytest.mark.parametrize('attr', ['edges', 'edges_out'])
def test_addEdge_is_idempotent(attr):
    start = Node('a')
    end = Node('b')
    edge = Edge(start, end)

    start.addEdge(edge)
    start.addEdge(edge)

    assert len(getattr(start, attr)) == 1
