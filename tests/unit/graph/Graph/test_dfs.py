"""Tests for Graph.dfs."""


def test_uses_the_directed_search_on_a_directed_graph(buildGraph):
    graph = buildGraph([('A', 'B'), ('B', 'C')])
    start = graph.getNodeByKey('C')

    result = [node.key for node in graph.dfs(start)]

    assert result == ['C']


def test_uses_the_undirected_search_on_an_undirected_graph(buildGraph):
    graph = buildGraph([('A', 'B')], is_directed=False)
    start = graph.getNodeByKey('B')

    result = [node.key for node in graph.dfs(start)]

    assert set(result) == {'A', 'B'}


def test_yields_nothing_for_a_none_start(buildGraph):
    graph = buildGraph([('A', 'B')])

    assert list(graph.dfs(None)) == []
