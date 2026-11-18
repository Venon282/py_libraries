"""Tests for Tree.getDescendants."""


def test_returns_the_descendants(tree):
    assert set(tree.getDescendants(tree.nodes['R'])) == {
        tree.nodes['A'],
        tree.nodes['B'],
        tree.nodes['C'],
    }
    assert tree.getDescendants(tree.nodes['A']) == [tree.nodes['C']]


def test_returns_an_empty_list_for_a_leaf(tree):
    assert tree.getDescendants(tree.nodes['B']) == []
