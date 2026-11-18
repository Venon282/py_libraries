"""Tests for the Graph.edgesWidth property."""

import numpy as np
from py_libraries.graph.Graph import Graph


def test_returns_the_widths_of_the_edges(buildGraph):
    graph = buildGraph([('A', 'B')])
    next(iter(graph.edges)).width = 0.5

    assert np.array_equal(graph.edgesWidth, [0.5])


def test_returns_an_empty_array_for_an_empty_graph():
    assert len(Graph().edgesWidth) == 0
