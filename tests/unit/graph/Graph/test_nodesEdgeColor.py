"""Tests for the Graph.nodesEdgeColor property."""

import numpy as np
from py_libraries.graph.Graph import Graph


def test_returns_the_border_colors_of_the_nodes(buildGraph):
    graph = buildGraph([('A', 'B')])
    graph.nodes['A'].edge_color = 'red'
    graph.nodes['B'].edge_color = 'blue'

    assert np.array_equal(graph.nodesEdgeColor, ['red', 'blue'])


def test_returns_an_empty_array_for_an_empty_graph():
    assert len(Graph().nodesEdgeColor) == 0
