"""Tests for Graph.dfsNonDirected."""


def test_explores_both_edge_directions(buildGraph):
    graph = buildGraph([('A', 'B')], is_directed=False)
    start = graph.getNodeByKey('B')

    result = [node.key for node in graph.dfsNonDirected(start)]

    assert set(result) == {'A', 'B'}
    assert result[0] == 'B'


def test_pre_and_post_order(buildGraph):
    graph = buildGraph([('A', 'B'), ('A', 'C')], is_directed=False)
    start = graph.getNodeByKey('A')

    pre = [node.key for node in graph.dfsNonDirected(start, order='pre')]
    post = [node.key for node in graph.dfsNonDirected(start, order='post')]

    assert pre[0] == 'A'
    assert post[-1] == 'A'
    assert set(pre) == set(post) == {'A', 'B', 'C'}


def test_yields_nothing_for_a_none_start(buildGraph):
    graph = buildGraph([('A', 'B')], is_directed=False)

    assert list(graph.dfsNonDirected(None)) == []


def test_does_not_revisit_nodes(buildGraph):
    graph = buildGraph([('A', 'B'), ('A', 'C'), ('B', 'C')], is_directed=False)
    start = graph.getNodeByKey('A')

    result = [node.key for node in graph.dfsNonDirected(start)]

    assert len(result) == len(set(result))
    assert set(result) == {'A', 'B', 'C'}
