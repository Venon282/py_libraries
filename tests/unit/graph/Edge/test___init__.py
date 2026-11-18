"""Tests for Edge.__init__."""

from py_libraries.graph.Edge import Edge
from py_libraries.graph.Node import Node


def test_init_stores_endpoints_and_weight():
    start = Node('a')
    end = Node('b')

    edge = Edge(start, end, weight=5)

    assert edge.start is start
    assert edge.end is end
    assert edge.weight == 5


def test_init_defaults_weight_properties_color_and_width():
    start = Node('a')
    end = Node('b')

    edge = Edge(start, end)

    assert edge.weight == 1
    assert edge.properties == {}
    assert edge.color == 'black'
    assert edge.width == 0.003


def test_init_properties_none_becomes_empty_dict():
    start = Node('a')
    end = Node('b')

    edge = Edge(start, end, properties=None)

    assert edge.properties == {}


def test_init_keeps_custom_properties():
    start = Node('a')
    end = Node('b')

    edge = Edge(start, end, properties={'side': 'left'})

    assert edge.properties == {'side': 'left'}
