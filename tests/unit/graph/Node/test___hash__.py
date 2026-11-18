"""Tests for Node.__hash__."""

from py_libraries.graph.Node import Node


def test_hash_equals_hash_of_key():
    node = Node('a')

    assert hash(node) == hash('a')


def test_nodes_with_same_key_hash_equal():
    assert hash(Node('a')) == hash(Node('a'))
