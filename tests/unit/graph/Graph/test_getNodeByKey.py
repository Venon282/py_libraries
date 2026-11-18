"""Tests for Graph.getNodeByKey."""


def test_returns_the_node_with_the_matching_key(buildGraph):
    graph = buildGraph([('A', 'B')])

    assert graph.getNodeByKey('A') is graph.nodes['A']


def test_returns_none_for_an_unknown_key(buildGraph):
    graph = buildGraph([('A', 'B')])

    assert graph.getNodeByKey('Z') is None
