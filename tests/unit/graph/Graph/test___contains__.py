"""Tests for Graph.__contains__."""

from py_libraries.graph.Node import Node


def test_contains_a_node_by_key(buildGraph):
    graph = buildGraph([('A', 'B')])

    assert Node('A') in graph
    assert Node('B') in graph


def test_does_not_contain_an_unknown_node(buildGraph):
    graph = buildGraph([('A', 'B')])

    assert Node('Z') not in graph


def test_contains_matches_by_key_not_identity(buildGraph):
    # A different Node object with the same key is still "in" the graph.
    graph = buildGraph([('A', 'B')])

    assert Node('A', value=999) in graph
