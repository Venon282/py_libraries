"""Tests for the Graph.nodesY property."""

import numpy as np
from py_libraries.graph.Graph import Graph


def test_returns_the_y_coordinates_of_the_nodes(buildGraph):
    graph = buildGraph([('A', 'B')])
    graph.nodes['A'].y = 2.0
    graph.nodes['B'].y = 4.0

    assert np.array_equal(graph.nodesY, [2.0, 4.0])


def test_returns_an_empty_array_for_an_empty_graph():
    assert len(Graph().nodesY) == 0
