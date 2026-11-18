"""Tests for Graph.addEdge."""

from py_libraries.graph.Edge import Edge
from py_libraries.graph.Graph import Graph
from py_libraries.graph.Node import Node


def test_adds_the_edge_and_registers_it_with_nodes():
    graph = Graph()
    a = Node('a')
    b = Node('b')
    edge = Edge(a, b)

    graph.addEdge(edge)

    assert edge in graph.edges
    assert edge in a.edges_out
    assert edge in b.edges_in


def test_adding_an_edge_adds_missing_nodes():
    graph = Graph()
    a = Node('a')
    b = Node('b')

    graph.addEdge(Edge(a, b))

    assert a in graph
    assert b in graph


def test_adding_the_same_edge_twice_keeps_one():
    graph = Graph()
    a = Node('a')
    b = Node('b')
    edge = Edge(a, b)

    graph.addEdge(edge)
    graph.addEdge(edge)

    assert len(graph.edges) == 1
