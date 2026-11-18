"""Tests for Graph.getEdgesPosition."""

import numpy as np
from py_libraries.graph.Graph import Graph


def test_returns_one_point_per_edge(buildGraph):
    graph = buildGraph([('A', 'B'), ('B', 'C')])

    (start_x, start_y), (end_x, end_y) = graph.getEdgesPosition()

    assert len(start_x) == len(graph.edges)
    assert len(start_y) == len(graph.edges)
    assert len(end_x) == len(graph.edges)
    assert len(end_y) == len(graph.edges)


def test_trims_endpoints_towards_each_other(buildGraph):
    graph = buildGraph([('A', 'B')])
    graph.nodes['A'].x = 0.0
    graph.nodes['A'].y = 0.0
    graph.nodes['B'].x = 10.0
    graph.nodes['B'].y = 0.0

    (start_x, start_y), (end_x, end_y) = graph.getEdgesPosition()

    # Each endpoint moves inward by the node radius along the edge direction.
    assert start_x[0] == np.float64(2.0)
    assert start_y[0] == np.float64(0.0)
    assert end_x[0] == np.float64(8.0)
    assert end_y[0] == np.float64(0.0)


def test_returns_empty_arrays_for_an_empty_graph():
    (start_x, start_y), (end_x, end_y) = Graph().getEdgesPosition()

    assert len(start_x) == 0
    assert len(start_y) == 0
    assert len(end_x) == 0
    assert len(end_y) == 0
