"""Tests for Tree.isLeaf."""


def test_leaves_are_recognized(tree):
    assert tree.isLeaf(tree.nodes['B']) is True
    assert tree.isLeaf(tree.nodes['C']) is True


def test_non_leaves_are_not_recognized(tree):
    assert tree.isLeaf(tree.nodes['R']) is False
    assert tree.isLeaf(tree.nodes['A']) is False
