"""Tests for Tree.isAnscestorOf."""


def test_ancestor_is_recognized(tree):
    assert tree.isAnscestorOf(tree.nodes['R'], tree.nodes['C']) is True
    assert tree.isAnscestorOf(tree.nodes['A'], tree.nodes['C']) is True


def test_non_ancestor_is_not_recognized(tree):
    assert tree.isAnscestorOf(tree.nodes['C'], tree.nodes['R']) is False
    assert tree.isAnscestorOf(tree.nodes['B'], tree.nodes['C']) is False
