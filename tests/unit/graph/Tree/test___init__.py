"""Tests for Tree.__init__."""

from py_libraries.graph.Tree import Tree


def test_is_directed_and_has_no_root():
    tree = Tree()

    assert tree.is_directed is True
    assert tree.root is None
    assert tree.nodes == {}


def test_stores_the_name():
    tree = Tree('mytree')

    assert tree.name == 'mytree'
