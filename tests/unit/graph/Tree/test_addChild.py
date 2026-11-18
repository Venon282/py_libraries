"""Tests for Tree.addChild."""

import pytest
from py_libraries.graph.Node import Node
from py_libraries.graph.Tree import Tree


def test_creates_an_edge_from_parent_to_child():
    tree = Tree()
    parent = Node('R')
    child = Node('A')
    tree.root = parent

    tree.addChild(parent, child)

    assert child in tree
    assert child.degree_in == 1
    assert parent.degree_out == 1


def test_raises_when_the_parent_is_not_in_the_tree():
    tree = Tree()
    parent = Node('R')
    child = Node('A')

    with pytest.raises(ValueError, match='not in the tree'):
        tree.addChild(parent, child)


def test_raises_when_the_child_is_already_in_the_tree(tree):
    with pytest.raises(ValueError, match='already in the tree'):
        tree.addChild(tree.nodes['R'], tree.nodes['A'])


def test_stores_the_edge_properties(tree):
    parent = tree.root
    child = Node('X')

    tree.addChild(parent, child, weight=5, properties={'side': 'left'})

    edge = next(iter(child.edges_in))
    assert edge.weight == 5
    assert edge.properties == {'side': 'left'}
