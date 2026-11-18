"""Tests for BinaryTree.addChild."""

import pytest
from py_libraries.graph.BinaryTree import BinaryTree
from py_libraries.graph.Node import Node


def test_addChild_always_raises():
    tree = BinaryTree()
    tree.root = Node('R')

    with pytest.raises(ValueError, match='use addLeft or addRight'):
        tree.addChild(tree.root, Node('C'))
