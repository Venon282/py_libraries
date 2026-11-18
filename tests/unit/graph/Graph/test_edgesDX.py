"""Tests for the Graph.edgesDX property."""

import numpy as np
from py_libraries.graph.Graph import Graph


def test_returns_the_horizontal_distance_of_the_edges(buildGraph):
    graph = buildGraph([('A', 'B')])
    graph.nodes['A'].x = 1.0
    graph.nodes['B'].x = 5.0

    assert np.array_equal(graph.edgesDX, [4.0])


def test_returns_an_empty_array_for_an_empty_graph():
    assert len(Graph().edgesDX) == 0
