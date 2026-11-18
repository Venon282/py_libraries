"""Tests for Tree.getIAncestors."""


def test_yields_the_ancestors_from_the_parent_to_the_root(tree):
    result = list(tree.getIAncestors(tree.nodes['C']))

    assert result == [tree.nodes['A'], tree.nodes['R']]


def test_yields_nothing_for_the_root(tree):
    assert list(tree.getIAncestors(tree.nodes['R'])) == []
