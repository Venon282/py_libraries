"""Tests for Graph.toList."""

from py_libraries.graph.Graph import Graph


def test_returns_one_pair_per_edge(buildGraph):
    graph = buildGraph([('A', 'B'), ('B', 'C')])

    result = graph.toList()

    assert sorted(result) == [('A', 'B'), ('B', 'C')]


def test_empty_graph_returns_an_empty_list():
    assert Graph().toList() == []
