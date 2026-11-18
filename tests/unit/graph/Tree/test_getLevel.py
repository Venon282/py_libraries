"""Tests for Tree.getLevel."""

from py_libraries.graph.Tree import Tree


def test_returns_the_nodes_at_the_given_depth(tree):
    assert tree.getLevel(0) == [tree.nodes['R']]
    assert set(tree.getLevel(1)) == {tree.nodes['A'], tree.nodes['B']}
    assert tree.getLevel(2) == [tree.nodes['C']]


def test_returns_an_empty_list_beyond_the_height(tree):
    assert tree.getLevel(5) == []


def test_returns_an_empty_list_when_there_is_no_root():
    assert Tree().getLevel(0) == []
