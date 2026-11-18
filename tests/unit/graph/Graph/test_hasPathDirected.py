"""Tests for Graph.hasPathDirected."""


def test_path_following_edge_direction(buildGraph):
    graph = buildGraph([('A', 'B'), ('B', 'C')])
    source = graph.getNodeByKey('A')
    target = graph.getNodeByKey('C')

    assert graph.hasPathDirected(source, target) is True


def test_no_path_against_edge_direction(buildGraph):
    graph = buildGraph([('A', 'B'), ('B', 'C')])
    source = graph.getNodeByKey('C')
    target = graph.getNodeByKey('A')

    assert graph.hasPathDirected(source, target) is False


def test_source_equals_target(buildGraph):
    graph = buildGraph([('A', 'B')])
    source = graph.getNodeByKey('A')

    assert graph.hasPathDirected(source, source) is True


def test_no_path_between_disconnected_nodes(buildGraph):
    graph = buildGraph([('A', 'B'), ('C', 'D')])
    source = graph.getNodeByKey('A')
    target = graph.getNodeByKey('D')

    assert graph.hasPathDirected(source, target) is False
