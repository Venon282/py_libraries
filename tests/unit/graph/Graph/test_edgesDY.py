"""Tests for the Graph.edgesDY property."""

import numpy as np
from py_libraries.graph.Graph import Graph


def test_returns_the_vertical_distance_of_the_edges(buildGraph):
    graph = buildGraph([('A', 'B')])
    graph.nodes['A'].y = 2.0
    graph.nodes['B'].y = 5.0

    assert np.array_equal(graph.edgesDY, [3.0])


def test_returns_an_empty_array_for_an_empty_graph():
    assert len(Graph().edgesDY) == 0
