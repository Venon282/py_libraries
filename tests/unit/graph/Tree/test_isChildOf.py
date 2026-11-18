"""Tests for Tree.isChildOf."""


def test_child_is_recognized(tree):
    assert tree.isChildOf(tree.nodes['A'], tree.nodes['R']) is True
    assert tree.isChildOf(tree.nodes['C'], tree.nodes['A']) is True


def test_non_child_is_not_recognized(tree):
    assert tree.isChildOf(tree.nodes['C'], tree.nodes['R']) is False
    assert tree.isChildOf(tree.nodes['R'], tree.nodes['A']) is False
