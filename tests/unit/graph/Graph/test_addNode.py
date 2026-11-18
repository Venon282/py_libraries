"""Tests for Graph.addNode."""

from py_libraries.graph.Edge import Edge
from py_libraries.graph.Graph import Graph
from py_libraries.graph.Node import Node


def test_adds_a_new_node():
    graph = Graph()
    node = Node('a')

    graph.addNode(node)

    assert graph.getNodeByKey('a') is node


def test_ignores_a_node_with_an_existing_key():
    graph = Graph()
    first = Node('a')
    second = Node('a')

    graph.addNode(first)
    graph.addNode(second)

    assert graph.getNodeByKey('a') is first
    assert len(graph) == 1


def test_adding_an_edge_also_adds_its_nodes():
    graph = Graph()
    a = Node('a')
    b = Node('b')

    graph.addEdge(Edge(a, b))

    assert a in graph
    assert b in graph
