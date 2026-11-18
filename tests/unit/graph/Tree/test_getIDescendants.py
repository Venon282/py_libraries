"""Tests for Tree.getIDescendants."""


def test_yields_the_descendants(tree):
    result = list(tree.getIDescendants(tree.nodes['R']))

    assert set(result) == {tree.nodes['A'], tree.nodes['B'], tree.nodes['C']}


def test_yields_nothing_for_a_leaf(tree):
    assert list(tree.getIDescendants(tree.nodes['B'])) == []
