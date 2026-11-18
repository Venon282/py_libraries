"""Tests for Graph.toDict."""

from py_libraries.graph.Graph import Graph


def test_contains_nodes_and_edges(buildGraph):
    graph = buildGraph([('A', 'B')])

    result = graph.toDict()

    assert set(result) == {'nodes', 'edges'}
    assert len(result['nodes']) == 2
    assert len(result['edges']) == 1


def test_nodes_are_serialized_with_their_toDict(buildGraph):
    graph = buildGraph([('A', 'B')])
    graph.nodes['A'].x = 1.0
    graph.nodes['A'].y = 2.0

    nodes = {node['key']: node for node in graph.toDict()['nodes']}

    assert nodes['A']['position'] == {'x': 1.0, 'y': 2.0}


def test_edges_are_serialized_with_their_toDict(buildGraph):
    graph = buildGraph([('A', 'B')])

    edges = graph.toDict()['edges']

    assert edges[0]['start'] == 'A'
    assert edges[0]['end'] == 'B'


def test_empty_graph_has_empty_lists():
    result = Graph().toDict()

    assert result == {'nodes': [], 'edges': []}
