"""Tests for Tree.isParentOf."""


def test_parent_is_recognized(tree):
    assert tree.isParentOf(tree.nodes['R'], tree.nodes['A']) is True
    assert tree.isParentOf(tree.nodes['A'], tree.nodes['C']) is True


def test_non_parent_is_not_recognized(tree):
    assert tree.isParentOf(tree.nodes['R'], tree.nodes['C']) is False
    assert tree.isParentOf(tree.nodes['A'], tree.nodes['R']) is False
