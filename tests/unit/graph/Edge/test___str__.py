"""Tests for Edge.__str__."""

from py_libraries.graph.Edge import Edge
from py_libraries.graph.Node import Node


def test_str_contains_endpoints_and_weight():
    edge = Edge(Node('a'), Node('b'), weight=5)

    text = str(edge)

    assert 'a -- b' in text
    assert 'Weight 5' in text


def test_str_contains_appearance():
    edge = Edge(Node('a'), Node('b'), color='red', width=0.5)

    text = str(edge)

    assert 'Color red' in text
    assert 'Width 0.5' in text
