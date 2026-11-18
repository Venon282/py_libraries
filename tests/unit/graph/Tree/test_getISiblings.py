"""Tests for Tree.getISiblings."""


def test_yields_the_siblings(tree):
    result = list(tree.getISiblings(tree.nodes['A']))

    assert set(result) == {tree.nodes['B']}


def test_yields_nothing_for_the_root(tree):
    assert list(tree.getISiblings(tree.nodes['R'])) == []
