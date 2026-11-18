"""Shared fixtures for the graph.Graph tests."""

import pytest
from py_libraries.graph.Edge import Edge
from py_libraries.graph.Graph import Graph
from py_libraries.graph.Node import Node


def build(pairs: list[tuple[str, str]], is_directed: bool = True) -> Graph:
    """Build a graph from (start_key, end_key) pairs."""
    graph = Graph(is_directed=is_directed)
    for start_key, end_key in pairs:
        start = graph.getNodeByKey(start_key) or Node(start_key)
        end = graph.getNodeByKey(end_key) or Node(end_key)
        graph.addEdge(Edge(start, end))
    return graph


@pytest.fixture
def buildGraph():
    """Provide a builder that creates a graph from (start_key, end_key) pairs."""
    return build


@pytest.fixture
def directed_line() -> Graph:
    """A directed chain A -> B -> C."""
    return build([('A', 'B'), ('B', 'C')])


@pytest.fixture
def undirected_triangle() -> Graph:
    """An undirected triangle A-B, B-C, A-C."""
    return build([('A', 'B'), ('B', 'C'), ('A', 'C')], is_directed=False)
