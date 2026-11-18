"""Tests for Node.__str__."""

from py_libraries.graph.Node import Node


def test_str_contains_key_value_and_position():
    node = Node('a', value=42, x=1.0, y=2.0, radius=3.0)

    text = str(node)

    assert 'a' in text
    assert '42' in text
    assert '(1.0, 2.0)' in text


def test_str_contains_appearance():
    node = Node('a', radius=3.0, color='red', edge_color='blue', edge_width=2.0)

    text = str(node)

    assert 'Diameter 6.0' in text
    assert 'Color red' in text
    assert 'Border blue' in text
    assert 'width 2.0' in text
