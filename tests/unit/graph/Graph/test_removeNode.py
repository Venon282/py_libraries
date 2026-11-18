"""Tests for Graph.removeNode."""

from py_libraries.graph.Edge import Edge
from py_libraries.graph.Graph import Graph
from py_libraries.graph.Node import Node


def test_removes_the_node():
    graph = Graph()
    a = Node('a')
    b = Node('b')
    edge = Edge(a, b)
    graph.addEdge(edge)

    graph.removeNode(a)

    assert a not in graph
    assert b in graph


def test_removes_connected_edges():
    graph = Graph()
    a = Node('a')
    b = Node('b')
    c = Node('c')
    ab = Edge(a, b)
    bc = Edge(b, c)
    graph.addEdge(ab)
    graph.addEdge(bc)

    graph.removeNode(b)

    assert len(graph.edges) == 0
    assert a.edges == set()
    assert c.edges == set()


def test_removing_an_unknown_node_is_a_noop():
    graph = Graph()
    a = Node('a')
    b = Node('b')
    edge = Edge(a, b)
    graph.addEdge(edge)

    graph.removeNode(Node('ghost'))

    assert a in graph
    assert b in graph
    assert len(graph.edges) == 1


def test_removes_self_loop():
    graph = Graph()
    a = Node('a')
    loop = Edge(a, a)
    graph.addEdge(loop)

    graph.removeNode(a)

    assert a not in graph
    assert len(graph.edges) == 0
