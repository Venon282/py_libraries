"""Tests for Graph.isLinkedList."""

from py_libraries.graph.Graph import Graph


def test_empty_graph_is_not_a_linked_list():
    assert Graph().isLinkedList() is False


def test_single_node_is_not_a_linked_list():
    graph = Graph()
    graph.addNode(Graph.constructByKeyPairs([('A', 'B')]).getNodeByKey('A'))

    assert graph.isLinkedList() is False


def test_single_node_with_a_self_loop_is_not_a_linked_list(buildGraph):
    graph = buildGraph([('A', 'A')])

    assert graph.isLinkedList() is False


def test_simple_chain_is_a_linked_list(buildGraph):
    graph = buildGraph([('A', 'B'), ('B', 'C')])

    assert graph.isLinkedList() is True


def test_fork_is_not_a_linked_list(buildGraph):
    graph = buildGraph([('A', 'B'), ('A', 'C')])

    assert graph.isLinkedList() is False


def test_cycle_is_not_a_linked_list(buildGraph):
    graph = buildGraph([('A', 'B'), ('B', 'C'), ('C', 'A')])

    assert graph.isLinkedList() is False


def test_two_heads_is_not_a_linked_list(buildGraph):
    graph = buildGraph([('A', 'C'), ('B', 'C')])

    assert graph.isLinkedList() is False


def test_disconnected_chain_is_not_a_linked_list(buildGraph):
    graph = buildGraph([('A', 'B'), ('C', 'D')])

    assert graph.isLinkedList() is False
