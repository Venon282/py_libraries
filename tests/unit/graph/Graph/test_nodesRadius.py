"""Tests for the Graph.nodesRadius property."""

import numpy as np
from py_libraries.graph.Graph import Graph


def test_returns_the_radii_of_the_nodes(buildGraph):
    graph = buildGraph([('A', 'B')])
    graph.nodes['A'].radius = 5.0
    graph.nodes['B'].radius = 7.0

    assert np.array_equal(graph.nodesRadius, [5.0, 7.0])


def test_returns_an_empty_array_for_an_empty_graph():
    assert len(Graph().nodesRadius) == 0
