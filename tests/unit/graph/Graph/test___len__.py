"""Tests for Graph.__len__."""

from py_libraries.graph.Graph import Graph


def test_len_is_the_number_of_nodes(buildGraph):
    graph = buildGraph([('A', 'B'), ('B', 'C')])

    assert len(graph) == 3


def test_len_is_zero_for_an_empty_graph():
    assert len(Graph()) == 0
