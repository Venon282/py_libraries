"""Tests for Graph.bfsDirected."""


def test_yields_nodes_in_breadth_first_order(buildGraph):
    graph = buildGraph([('A', 'B'), ('A', 'C'), ('B', 'D')])
    start = graph.getNodeByKey('A')

    result = [node.key for node in graph.bfsDirected(start)]

    assert result[0] == 'A'
    assert set(result) == {'A', 'B', 'C', 'D'}
    # A node is always visited before its successors.
    assert result.index('A') < result.index('B')
    assert result.index('A') < result.index('C')
    assert result.index('B') < result.index('D')


def test_yields_nothing_for_a_none_start(buildGraph):
    graph = buildGraph([('A', 'B')])

    assert list(graph.bfsDirected(None)) == []


def test_does_not_revisit_nodes(buildGraph):
    graph = buildGraph([('A', 'B'), ('A', 'C'), ('B', 'C')])
    start = graph.getNodeByKey('A')

    result = [node.key for node in graph.bfsDirected(start)]

    assert len(result) == len(set(result))
    assert set(result) == {'A', 'B', 'C'}


def test_reuses_the_provided_visited_set(buildGraph):
    graph = buildGraph([('A', 'B'), ('B', 'C')])
    start = graph.getNodeByKey('A')
    visited: set[str] = set()

    list(graph.bfsDirected(start, visited=visited))

    assert visited == {'A', 'B', 'C'}


def test_skips_nodes_already_in_visited(buildGraph):
    graph = buildGraph([('A', 'B')])
    start = graph.getNodeByKey('A')

    result = [node.key for node in graph.bfsDirected(start, visited={'A'})]

    assert result == []
