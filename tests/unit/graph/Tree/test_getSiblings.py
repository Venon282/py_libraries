"""Tests for Tree.getSiblings."""


def test_returns_the_siblings(tree):
    assert set(tree.getSiblings(tree.nodes['A'])) == {tree.nodes['B']}
    assert set(tree.getSiblings(tree.nodes['B'])) == {tree.nodes['A']}


def test_returns_an_empty_list_for_the_root(tree):
    assert tree.getSiblings(tree.nodes['R']) == []


def test_excludes_the_node_itself(tree):
    siblings = tree.getSiblings(tree.nodes['A'])

    assert tree.nodes['A'] not in siblings
