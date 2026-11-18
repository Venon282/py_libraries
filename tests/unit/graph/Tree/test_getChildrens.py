"""Tests for Tree.getChildrens."""


def test_returns_the_children(tree):
    assert set(tree.getChildrens(tree.nodes['R'])) == {
        tree.nodes['A'],
        tree.nodes['B'],
    }
    assert tree.getChildrens(tree.nodes['A']) == [tree.nodes['C']]


def test_returns_an_empty_list_for_a_leaf(tree):
    assert tree.getChildrens(tree.nodes['B']) == []
