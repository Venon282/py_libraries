"""Tests for Tree.getLeaves."""

from py_libraries.graph.Node import Node
from py_libraries.graph.Tree import Tree


def test_returns_the_nodes_without_children(tree):
    assert set(tree.getLeaves()) == {tree.nodes['B'], tree.nodes['C']}


def test_a_single_node_tree_has_one_leaf():
    tree = Tree()
    r = Node('R')
    tree.root = r

    assert tree.getLeaves() == [r]
