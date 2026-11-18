"""Tests for the Tree.root property."""

import pytest
from py_libraries.graph.Edge import Edge
from py_libraries.graph.Node import Node
from py_libraries.graph.Tree import Tree


def test_root_is_none_by_default():
    assert Tree().root is None


def test_setting_the_root_adds_it_to_the_tree():
    tree = Tree()
    root = Node('R')

    tree.root = root

    assert tree.root is root
    assert root in tree


def test_setting_the_root_raises_when_the_node_has_parents():
    tree = Tree()
    parent = Node('P')
    child = Node('C')
    tree.addEdge(Edge(parent, child))

    with pytest.raises(ValueError, match="can't be a root"):
        tree.root = child
