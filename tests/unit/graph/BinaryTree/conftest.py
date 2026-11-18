"""Shared fixtures for the BinaryTree tests."""

import pytest
from py_libraries.graph.BinaryTree import BinaryTree
from py_libraries.graph.Node import Node


def buildBinaryTree() -> BinaryTree:
    """Build a binary tree R with left L and right R.

    Returns:
        The built binary tree.
    """
    tree = BinaryTree()
    root = Node('R')
    tree.root = root
    tree.addLeft(root, Node('L'))
    tree.addRight(root, Node('R2'))

    return tree


@pytest.fixture
def binary_tree() -> BinaryTree:
    """A binary tree with root R, left child L and right child R2."""
    return buildBinaryTree()
