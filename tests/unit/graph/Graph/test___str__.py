"""Tests for Graph.__str__."""


def test_str_contains_name_nodes_and_edges(buildGraph):
    graph = buildGraph([('A', 'B')])

    text = str(graph)

    assert '--- Graph undefined ---' in text
    assert 'Nodes:' in text
    assert 'Edges:' in text
    assert 'A' in text
    assert 'B' in text


def test_str_uses_arrows_for_a_directed_graph(buildGraph):
    graph = buildGraph([('A', 'B')])

    assert 'A->B' in str(graph)


def test_str_uses_dashes_for_an_undirected_graph(buildGraph):
    graph = buildGraph([('A', 'B')], is_directed=False)

    assert 'A-B' in str(graph)


def test_str_shows_the_graph_name(buildGraph):
    graph = buildGraph([('A', 'B')])
    graph.name = 'mygraph'

    assert '--- Graph mygraph ---' in str(graph)
