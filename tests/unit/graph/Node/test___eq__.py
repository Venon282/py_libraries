"""Tests for Node.__eq__."""

from py_libraries.graph.Node import Node


def test_equal_nodes_share_the_same_key():
    assert Node('a') == Node('a')


def test_nodes_with_different_keys_are_not_equal():
    assert Node('a') != Node('b')


def test_equality_ignores_other_attributes():
    a = Node('a', value=1, x=0.0, y=0.0)
    b = Node('a', value=2, x=9.0, y=9.0)

    assert a == b


def test_not_equal_to_a_non_node():
    assert (Node('a') == 'a') is False
    assert (Node('a') == 1) is False
