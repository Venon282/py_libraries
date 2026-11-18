"""Tests for Node.toDict."""

from py_libraries.graph.Node import Node


def test_toDict_contains_all_fields():
    node = Node('a', value=42, x=1.0, y=2.0, radius=3.0)

    result = node.toDict()

    assert result['key'] == 'a'
    assert result['value'] == 42
    assert result['position'] == {'x': 1.0, 'y': 2.0}
    assert result['radius'] == 3.0
    assert result['diameter'] == 6.0
    assert result['color'] == 'white'
    assert result['edge_color'] == 'black'
    assert result['edge_width'] == 1.0
    assert result['properties'] == {}


def test_toDict_reflects_custom_properties():
    node = Node('a', properties={'kind': 'leaf'})

    assert node.toDict()['properties'] == {'kind': 'leaf'}
