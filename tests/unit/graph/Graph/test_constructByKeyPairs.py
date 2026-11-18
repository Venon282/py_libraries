"""Tests for Graph.constructByKeyPairs."""

from py_libraries.graph.Graph import Graph


def test_constructs_nodes_and_edges():
    graph = Graph.constructByKeyPairs([('A', 'B'), ('B', 'C')])

    assert set(graph.nodes) == {'A', 'B', 'C'}
    assert len(graph.edges) == 2


def test_is_directed_by_default():
    graph = Graph.constructByKeyPairs([('A', 'B')])

    assert graph.is_directed is True


def test_shares_nodes_across_pairs():
    graph = Graph.constructByKeyPairs([('A', 'B'), ('A', 'C')])

    # A appears in two pairs but must be a single node.
    assert len(graph.nodes) == 3
    assert graph.getNodeByKey('A') is graph.nodes['A']


def test_registers_edges_with_nodes():
    graph = Graph.constructByKeyPairs([('A', 'B')])

    a = graph.getNodeByKey('A')
    b = graph.getNodeByKey('B')
    assert a.degree_out == 1
    assert b.degree_in == 1


def test_converts_keys_to_strings():
    graph = Graph.constructByKeyPairs([(1, 2)])

    assert set(graph.nodes) == {'1', '2'}


def test_empty_pairs_gives_empty_graph():
    graph = Graph.constructByKeyPairs([])

    assert len(graph) == 0
    assert len(graph.edges) == 0
