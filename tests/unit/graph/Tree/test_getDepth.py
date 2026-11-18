"""Tests for Tree.getDepth."""

import pytest
from py_libraries.graph.Node import Node
from py_libraries.graph.Tree import Tree


def test_root_is_at_depth_zero(tree):
    assert tree.getDepth(tree.nodes['R']) == 0


def test_depth_is_the_distance_from_the_root(tree):
    assert tree.getDepth(tree.nodes['A']) == 1
    assert tree.getDepth(tree.nodes['C']) == 2


def test_raises_when_there_is_no_root():
    tree = Tree()
    orphan = Node('A')
    tree.addNode(orphan)

    with pytest.raises(ValueError, match='root is not defined'):
        tree.getDepth(orphan)
