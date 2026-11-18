"""Tests for Tree.getAncestors."""


def test_returns_the_ancestors_from_the_parent_to_the_root(tree):
    assert tree.getAncestors(tree.nodes['C']) == [
        tree.nodes['A'],
        tree.nodes['R'],
    ]


def test_returns_an_empty_list_for_the_root(tree):
    assert tree.getAncestors(tree.nodes['R']) == []
