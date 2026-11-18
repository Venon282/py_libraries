"""Tests for Graph.dfsDirected."""


def test_pre_order_visits_node_before_successors(buildGraph):
    graph = buildGraph([('A', 'B'), ('A', 'C')])
    start = graph.getNodeByKey('A')

    result = [node.key for node in graph.dfsDirected(start, order='pre')]

    assert result[0] == 'A'
    assert set(result) == {'A', 'B', 'C'}
    assert result.index('A') < result.index('B')
    assert result.index('A') < result.index('C')


def test_post_order_visits_node_after_successors(buildGraph):
    graph = buildGraph([('A', 'B'), ('A', 'C')])
    start = graph.getNodeByKey('A')

    result = [node.key for node in graph.dfsDirected(start, order='post')]

    assert result[-1] == 'A'
    assert set(result) == {'A', 'B', 'C'}
    assert result.index('B') < result.index('A')
    assert result.index('C') < result.index('A')


def test_yields_nothing_for_a_none_start(buildGraph):
    graph = buildGraph([('A', 'B')])

    assert list(graph.dfsDirected(None)) == []


def test_does_not_revisit_nodes_in_a_cycle(buildGraph):
    graph = buildGraph([('A', 'B'), ('B', 'C'), ('C', 'A')])
    start = graph.getNodeByKey('A')

    result = [node.key for node in graph.dfsDirected(start)]

    assert len(result) == len(set(result))
    assert set(result) == {'A', 'B', 'C'}


def test_reuses_the_provided_visited_set(buildGraph):
    graph = buildGraph([('A', 'B')])
    start = graph.getNodeByKey('A')
    visited: set[str] = set()

    list(graph.dfsDirected(start, visited=visited))

    assert visited == {'A', 'B'}
