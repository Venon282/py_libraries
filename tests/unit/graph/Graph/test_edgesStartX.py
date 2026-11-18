"""Tests for the Graph.edgesStartX property."""

import numpy as np
from py_libraries.graph.Graph import Graph


def test_returns_the_x_coordinates_of_the_start_nodes(buildGraph):
    graph = buildGraph([('A', 'B')])
    graph.nodes['A'].x = 1.0
    graph.nodes['B'].x = 3.0

    assert np.array_equal(graph.edgesStartX, [1.0])


def test_returns_an_empty_array_for_an_empty_graph():
    assert len(Graph().edgesStartX) == 0
