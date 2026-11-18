"""Tests for BinaryTree.getChildren."""

from py_libraries.graph.BinaryTree import BinaryTree
from py_libraries.graph.Node import Node


def test_returns_left_then_right(binary_tree):
    assert binary_tree.getChildren(binary_tree.root) == [
        binary_tree.nodes['L'],
        binary_tree.nodes['R2'],
    ]


def test_returns_only_the_left_child():
    tree = BinaryTree()
    root = Node('R')
    tree.root = root
    tree.addLeft(root, Node('L'))

    assert tree.getChildren(root) == [tree.nodes['L']]


def test_returns_only_the_right_child():
    tree = BinaryTree()
    root = Node('R')
    tree.root = root
    tree.addRight(root, Node('R2'))

    assert tree.getChildren(root) == [tree.nodes['R2']]


def test_returns_an_empty_list_for_a_leaf(binary_tree):
    assert binary_tree.getChildren(binary_tree.nodes['L']) == []
