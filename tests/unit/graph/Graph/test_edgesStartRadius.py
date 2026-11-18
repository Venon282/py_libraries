"""Tests for the Graph.edgesStartRadius property."""

import numpy as np
from py_libraries.graph.Graph import Graph


def test_returns_the_radii_of_the_start_nodes(buildGraph):
    graph = buildGraph([('A', 'B')])
    graph.nodes['A'].radius = 5.0
    graph.nodes['B'].radius = 7.0

    assert np.array_equal(graph.edgesStartRadius, [5.0])


def test_returns_an_empty_array_for_an_empty_graph():
    assert len(Graph().edgesStartRadius) == 0
