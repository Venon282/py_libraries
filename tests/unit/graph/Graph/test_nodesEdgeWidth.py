"""Tests for the Graph.nodesEdgeWidth property."""

import numpy as np
from py_libraries.graph.Graph import Graph


def test_returns_the_border_widths_of_the_nodes(buildGraph):
    graph = buildGraph([('A', 'B')])
    graph.nodes['A'].edge_width = 1.5
    graph.nodes['B'].edge_width = 2.5

    assert np.array_equal(graph.nodesEdgeWidth, [1.5, 2.5])


def test_returns_an_empty_array_for_an_empty_graph():
    assert len(Graph().nodesEdgeWidth) == 0
