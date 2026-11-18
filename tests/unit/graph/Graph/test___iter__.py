"""Tests for Graph.__iter__."""

from py_libraries.graph.Graph import Graph


def test_iterates_over_all_nodes(buildGraph):
    graph = buildGraph([('A', 'B'), ('B', 'C')])

    keys = {node.key for node in graph}

    assert keys == {'A', 'B', 'C'}


def test_iterates_over_nothing_for_an_empty_graph():
    assert list(Graph()) == []
