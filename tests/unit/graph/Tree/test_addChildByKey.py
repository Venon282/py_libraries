"""Tests for Tree.addChildByKey."""

import pytest
from py_libraries.graph.Node import Node
from py_libraries.graph.Tree import Tree


def test_creates_the_child_node_if_missing():
    tree = Tree()
    tree.root = Node('R')

    tree.addChildByKey('R', 'A')

    assert tree.getNodeByKey('A') is not None
    assert tree.getNodeByKey('A').degree_in == 1


def test_raises_when_the_child_key_is_already_in_the_tree():
    tree = Tree()
    tree.root = Node('R')
    existing = Node('A')
    tree.addChild(tree.root, existing)

    with pytest.raises(ValueError, match='already in the tree'):
        tree.addChildByKey('R', 'A')

    assert tree.getNodeByKey('A') is existing


def test_raises_when_the_parent_key_is_unknown():
    tree = Tree()

    with pytest.raises(ValueError, match='No node found'):
        tree.addChildByKey('R', 'A')
