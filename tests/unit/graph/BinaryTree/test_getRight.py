"""Tests for BinaryTree.getRight."""

from py_libraries.graph.BinaryTree import BinaryTree
from py_libraries.graph.Node import Node


def test_returns_the_right_child(binary_tree):
    assert binary_tree.getRight(binary_tree.root) is binary_tree.nodes['R2']


def test_returns_none_when_there_is_no_right_child(binary_tree):
    assert binary_tree.getRight(binary_tree.nodes['R2']) is None


def test_returns_none_for_a_leaf():
    tree = BinaryTree()
    root = Node('R')
    tree.root = root

    assert tree.getRight(root) is None
