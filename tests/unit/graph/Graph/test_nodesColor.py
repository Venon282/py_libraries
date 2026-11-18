"""Tests for the Graph.nodesColor property."""

import numpy as np
from py_libraries.graph.Graph import Graph


def test_returns_the_fill_colors_of_the_nodes(buildGraph):
    graph = buildGraph([('A', 'B')])
    graph.nodes['A'].color = 'red'
    graph.nodes['B'].color = 'blue'

    assert np.array_equal(graph.nodesColor, ['red', 'blue'])


def test_returns_an_empty_array_for_an_empty_graph():
    assert len(Graph().nodesColor) == 0
