"""Tests for Graph.isRegularGrap."""

from py_libraries.graph.Graph import Graph


def test_empty_graph_is_regular():
    assert Graph().isRegularGrap() is True


def test_all_nodes_same_degree_is_regular(buildGraph):
    # A cycle A -> B -> C -> A: every node has in-degree 1 and out-degree 1.
    graph = buildGraph([('A', 'B'), ('B', 'C'), ('C', 'A')])

    assert graph.isRegularGrap() is True


def test_mixed_degrees_is_not_regular(buildGraph):
    graph = buildGraph([('A', 'B'), ('A', 'C')])

    assert graph.isRegularGrap() is False


def test_different_degrees_across_nodes_is_not_regular(buildGraph):
    # A -> B and C -> B: B has in-degree 2 and out-degree 0, the others differ.
    graph = buildGraph([('A', 'B'), ('C', 'B')])

    assert graph.isRegularGrap() is False
