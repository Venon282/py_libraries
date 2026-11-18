"""Tests for Graph.hasPath."""


def test_directed_graph_uses_directed_search(buildGraph):
    graph = buildGraph([('A', 'B'), ('B', 'C')])
    source = graph.getNodeByKey('C')
    target = graph.getNodeByKey('A')

    # No directed path from C to A.
    assert graph.hasPath(source, target) is False


def test_undirected_graph_uses_undirected_search(buildGraph):
    graph = buildGraph([('A', 'B'), ('B', 'C')], is_directed=False)
    source = graph.getNodeByKey('C')
    target = graph.getNodeByKey('A')

    # A path exists when direction is ignored.
    assert graph.hasPath(source, target) is True


def test_source_equals_target(buildGraph):
    graph = buildGraph([('A', 'B')])
    source = graph.getNodeByKey('A')

    assert graph.hasPath(source, source) is True
