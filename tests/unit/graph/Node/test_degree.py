"""Tests for the Node.degree_in and Node.degree_out properties."""

from py_libraries.graph.Edge import Edge
from py_libraries.graph.Node import Node


def test_degrees_are_zero_for_a_fresh_node():
    node = Node('a')

    assert node.degree_in == 0
    assert node.degree_out == 0


def test_degrees_count_connected_edges():
    a = Node('a')
    b = Node('b')
    c = Node('c')

    ab = Edge(a, b)
    bc = Edge(b, c)
    ca = Edge(c, a)

    for edge in (ab, bc, ca):
        edge.start.addEdge(edge)
        edge.end.addEdge(edge)

    assert a.degree_out == 1
    assert a.degree_in == 1
    assert b.degree_out == 1
    assert b.degree_in == 1
    assert c.degree_out == 1
    assert c.degree_in == 1
