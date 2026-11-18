"""Tests for Tree.getParent."""


def test_returns_the_parent(tree):
    assert tree.getParent(tree.nodes['A']) is tree.nodes['R']
    assert tree.getParent(tree.nodes['C']) is tree.nodes['A']


def test_returns_none_for_the_root(tree):
    assert tree.getParent(tree.nodes['R']) is None
