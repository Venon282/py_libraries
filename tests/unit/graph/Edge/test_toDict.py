"""Tests for Edge.toDict."""

from py_libraries.graph.Edge import Edge
from py_libraries.graph.Node import Node


def test_toDict_contains_endpoints_weight_and_properties():
    start = Node('a')
    end = Node('b')

    edge = Edge(start, end, weight=7, properties={'side': 'left'})

    result = edge.toDict()

    assert result == {
        'start': 'a',
        'end': 'b',
        'weight': 7,
        'properties': {'side': 'left'},
    }


def test_toDict_defaults():
    start = Node('a')
    end = Node('b')

    result = Edge(start, end).toDict()

    assert result['weight'] == 1
    assert result['properties'] == {}
