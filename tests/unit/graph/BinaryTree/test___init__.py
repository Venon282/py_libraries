"""Tests for BinaryTree.__init__."""

from py_libraries.graph.BinaryTree import BinaryTree


def test_empty_tree_has_no_root():
    tree = BinaryTree()

    assert tree.root is None
    assert len(tree) == 0


def test_default_name_is_undefined():
    assert BinaryTree().name == 'undefined'


def test_name_can_be_set():
    assert BinaryTree('bt').name == 'bt'


def test_is_directed():
    assert BinaryTree().is_directed is True
