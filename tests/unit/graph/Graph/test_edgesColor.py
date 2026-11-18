"""Tests for the Graph.edgesColor property."""

import numpy as np
from py_libraries.graph.Graph import Graph


def test_returns_the_colors_of_the_edges(buildGraph):
    graph = buildGraph([('A', 'B')])
    next(iter(graph.edges)).color = 'red'

    assert np.array_equal(graph.edgesColor, ['red'])


def test_returns_an_empty_array_for_an_empty_graph():
    assert len(Graph().edgesColor) == 0
