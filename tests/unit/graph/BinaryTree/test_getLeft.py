"""Tests for BinaryTree.getLeft."""

from py_libraries.graph.BinaryTree import BinaryTree
from py_libraries.graph.Node import Node


def test_returns_the_left_child(binary_tree):
    assert binary_tree.getLeft(binary_tree.root) is binary_tree.nodes['L']


def test_returns_none_when_there_is_no_left_child(binary_tree):
    assert binary_tree.getLeft(binary_tree.nodes['L']) is None


def test_returns_none_for_a_leaf():
    tree = BinaryTree()
    root = Node('R')
    tree.root = root

    assert tree.getLeft(root) is None
