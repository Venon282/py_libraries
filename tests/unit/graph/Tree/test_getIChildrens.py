"""Tests for Tree.getIChildrens."""


def test_yields_the_children(tree):
    result = list(tree.getIChildrens(tree.nodes['R']))

    assert set(result) == {tree.nodes['A'], tree.nodes['B']}


def test_yields_nothing_for_a_leaf(tree):
    assert list(tree.getIChildrens(tree.nodes['B'])) == []
