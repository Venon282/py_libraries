"""Tests for Node.__init__."""

import pytest
from py_libraries.graph.Node import Node


def test_init_stores_key_value_and_position():
    node = Node('a', value=42, x=1.0, y=2.0)

    assert node.key == 'a'
    assert node.value == 42
    assert node.x == 1.0
    assert node.y == 2.0


def test_init_converts_key_to_string():
    node = Node(123)

    assert node.key == '123'


def test_init_defaults_key_and_properties():
    node = Node()

    assert node.key == 'undefined'
    assert node.value is None
    assert node.properties == {}


def test_init_random_position_is_in_range():
    node = Node('a')

    assert 0.0 <= node.x < 100.0
    assert 0.0 <= node.y < 100.0


def test_init_diameter_is_twice_radius():
    node = Node('a', radius=3.0)

    assert node.diameter == 6.0


def test_init_properties_none_becomes_empty_dict():
    node = Node('a', properties=None)

    assert node.properties == {}


@pytest.mark.parametrize('radius', [0.5, 2.0, 10.0])
def test_init_appearance_defaults(radius: float):
    node = Node('a', radius=radius)

    assert node.radius == radius
    assert node.color == 'white'
    assert node.edge_color == 'black'
    assert node.edge_width == 1.0
