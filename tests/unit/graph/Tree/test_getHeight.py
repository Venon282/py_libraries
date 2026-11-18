"""Tests for Tree.getHeight."""

from py_libraries.graph.Tree import Tree


def test_height_of_the_tree(tree):
    # R -> A -> C is the longest path, so the height is 2.
    assert tree.getHeight() == 2


def test_height_of_a_subtree(tree):
    assert tree.getHeight(tree.nodes['A']) == 1
    assert tree.getHeight(tree.nodes['C']) == 0


def test_height_is_minus_one_when_there_is_no_root():
    assert Tree().getHeight() == -1
