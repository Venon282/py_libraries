"""Tests for the Graph.edgesCoordinate property."""

import numpy as np
from py_libraries.graph.Graph import Graph


def test_returns_the_trimmed_endpoints_of_the_edges(buildGraph):
    graph = buildGraph([('A', 'B')])
    graph.nodes['A'].x = 0.0
    graph.nodes['A'].y = 0.0
    graph.nodes['B'].x = 10.0
    graph.nodes['B'].y = 0.0

    coordinates = graph.edgesCoordinate

    assert coordinates.shape == (1, 2, 2)
    assert np.allclose(coordinates[0][0], [2.0, 0.0])
    assert np.allclose(coordinates[0][1], [8.0, 0.0])


def test_returns_an_empty_array_for_an_empty_graph():
    assert len(Graph().edgesCoordinate) == 0
