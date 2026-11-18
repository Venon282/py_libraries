"""Tests for Graph.isTree."""

from py_libraries.graph.Graph import Graph


def test_empty_graph_is_not_a_tree():
    assert Graph().isTree() is False


def test_single_node_is_a_tree():
    graph = Graph()
    graph.addNode(Graph.constructByKeyPairs([('A', 'B')]).getNodeByKey('A'))

    assert graph.isTree() is True


def test_directed_tree_is_a_tree(buildGraph):
    graph = buildGraph([('A', 'B'), ('A', 'C'), ('B', 'D')])

    assert graph.isTree() is True


def test_cycle_is_not_a_tree(buildGraph):
    graph = buildGraph([('A', 'B'), ('B', 'C'), ('C', 'A')])

    assert graph.isTree() is False


def test_node_with_two_parents_is_not_a_tree(buildGraph):
    graph = buildGraph([('A', 'C'), ('B', 'C')])

    assert graph.isTree() is False


def test_two_roots_is_not_a_tree(buildGraph):
    graph = buildGraph([('A', 'B'), ('C', 'D')])

    assert graph.isTree() is False


def test_unreachable_node_is_not_a_tree(buildGraph):
    # A -> B, plus a disconnected node C.
    graph = buildGraph([('A', 'B')])
    graph.addNode(Graph.constructByKeyPairs([('C', 'D')]).getNodeByKey('C'))

    assert graph.isTree() is False
