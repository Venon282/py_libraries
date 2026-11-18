"""Tests for Graph.hasPathNonDirected."""


def test_path_ignores_edge_direction(buildGraph):
    # A -> B, but the path from B to A must be found.
    graph = buildGraph([('A', 'B'), ('B', 'C')])
    source = graph.getNodeByKey('C')
    target = graph.getNodeByKey('A')

    assert graph.hasPathNonDirected(source, target) is True


def test_source_equals_target(buildGraph):
    graph = buildGraph([('A', 'B')])
    source = graph.getNodeByKey('A')

    assert graph.hasPathNonDirected(source, source) is True


def test_no_path_between_disconnected_nodes(buildGraph):
    graph = buildGraph([('A', 'B'), ('C', 'D')])
    source = graph.getNodeByKey('A')
    target = graph.getNodeByKey('D')

    assert graph.hasPathNonDirected(source, target) is False
