"""Tests for Graph.bfsNonDirected."""


def test_explores_both_edge_directions(buildGraph):
    # A -> B, but the search must be able to go from B back to A.
    graph = buildGraph([('A', 'B')], is_directed=False)
    start = graph.getNodeByKey('B')

    result = [node.key for node in graph.bfsNonDirected(start)]

    assert set(result) == {'A', 'B'}
    assert result[0] == 'B'


def test_yields_nothing_for_a_none_start(buildGraph):
    graph = buildGraph([('A', 'B')], is_directed=False)

    assert list(graph.bfsNonDirected(None)) == []


def test_does_not_revisit_nodes(buildGraph):
    graph = buildGraph([('A', 'B'), ('A', 'C'), ('B', 'C')], is_directed=False)
    start = graph.getNodeByKey('A')

    result = [node.key for node in graph.bfsNonDirected(start)]

    assert len(result) == len(set(result))
    assert set(result) == {'A', 'B', 'C'}


def test_reuses_the_provided_visited_set(buildGraph):
    graph = buildGraph([('A', 'B')], is_directed=False)
    start = graph.getNodeByKey('A')
    visited: set[str] = set()

    list(graph.bfsNonDirected(start, visited=visited))

    assert visited == {'A', 'B'}
