"""Tests for the Node.neighbours property."""

from py_libraries.graph.Edge import Edge
from py_libraries.graph.Node import Node


def test_neighbours_is_empty_for_a_fresh_node():
    node = Node('a')

    assert node.neighbours == set()


def test_neighbours_returns_adjacent_nodes_in_both_directions():
    a = Node('a')
    b = Node('b')
    c = Node('c')

    ab = Edge(a, b)
    ca = Edge(c, a)
    a.addEdge(ab)
    b.addEdge(ab)
    c.addEdge(ca)
    a.addEdge(ca)

    assert a.neighbours == {b, c}


def test_neighbours_excludes_self_loops():
    node = Node('a')
    loop = Edge(node, node)
    node.addEdge(loop)

    assert node.neighbours == set()
