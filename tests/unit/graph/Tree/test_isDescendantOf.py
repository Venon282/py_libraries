"""Tests for Tree.isDescendantOf."""


def test_descendant_is_recognized(tree):
    assert tree.isDescendantOf(tree.nodes['C'], tree.nodes['R']) is True
    assert tree.isDescendantOf(tree.nodes['C'], tree.nodes['A']) is True


def test_non_descendant_is_not_recognized(tree):
    assert tree.isDescendantOf(tree.nodes['R'], tree.nodes['C']) is False
    assert tree.isDescendantOf(tree.nodes['C'], tree.nodes['B']) is False
