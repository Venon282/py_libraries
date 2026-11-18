"""Tests for Tree.isSiblingOf."""


def test_siblings_are_recognized(tree):
    assert tree.isSiblingOf(tree.nodes['A'], tree.nodes['B']) is True
    assert tree.isSiblingOf(tree.nodes['B'], tree.nodes['A']) is True


def test_non_siblings_are_not_recognized(tree):
    # A and C are parent and child, not siblings.
    assert tree.isSiblingOf(tree.nodes['A'], tree.nodes['C']) is False
    # R has no siblings.
    assert tree.isSiblingOf(tree.nodes['R'], tree.nodes['A']) is False
