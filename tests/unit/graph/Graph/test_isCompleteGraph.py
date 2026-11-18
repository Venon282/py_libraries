"""Tests for Graph.isCompleteGraph."""

from py_libraries.graph.Graph import Graph


def test_single_node_is_complete():
    graph = Graph()
    graph.addNode(Graph.constructByKeyPairs([('A', 'B')]).getNodeByKey('A'))

    assert graph.isCompleteGraph() is True


def test_empty_graph_is_complete():
    assert Graph().isCompleteGraph() is True


def test_complete_graph_is_complete(buildGraph):
    # Every node points to every other node.
    graph = buildGraph(
        [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]
    )

    assert graph.isCompleteGraph() is True


def test_missing_edge_is_not_complete(buildGraph):
    # C is missing the edge to A.
    graph = buildGraph([('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A')])

    assert graph.isCompleteGraph() is False


def test_self_loop_is_not_complete(buildGraph):
    # A points to itself instead of B.
    graph = buildGraph(
        [('A', 'A'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]
    )

    assert graph.isCompleteGraph() is False
