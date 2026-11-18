"""Tests for Tree.isValidTree."""

from py_libraries.graph.Edge import Edge
from py_libraries.graph.Node import Node
from py_libraries.graph.Tree import Tree


def test_valid_tree(tree):
    assert tree.isValidTree() is True


def test_invalid_when_there_is_no_root():
    tree = Tree()

    assert tree.isValidTree() is False


def test_invalid_when_the_edge_count_is_wrong():
    # A root with two children but only one edge.
    tree = Tree()
    r = Node('R')
    a = Node('A')
    b = Node('B')
    tree.root = r
    tree.addChild(r, a)
    tree.addNode(b)

    assert tree.isValidTree() is False


def test_invalid_when_a_node_is_unreachable():
    tree = Tree()
    r = Node('R')
    a = Node('A')
    b = Node('B')
    tree.root = r
    tree.addEdge(Edge(r, a))
    tree.addEdge(Edge(b, a))
    # b is not reachable from r.

    assert tree.isValidTree() is False
