"""Tests for Graph.__init__."""

from py_libraries.graph.Graph import Graph


def test_init_defaults():
    graph = Graph()

    assert graph.name == 'undefined'
    assert graph.is_directed is True
    assert graph.nodes == {}
    assert graph.edges == set()
    assert graph.show_labels_max_nodes == 150


def test_init_stores_arguments():
    graph = Graph(name='g', is_directed=False, show_labels_max_nodes=5)

    assert graph.name == 'g'
    assert graph.is_directed is False
    assert graph.show_labels_max_nodes == 5
