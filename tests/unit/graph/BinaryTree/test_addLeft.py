"""Tests for BinaryTree.addLeft."""

import pytest
from py_libraries.graph.BinaryTree import BinaryTree
from py_libraries.graph.Node import Node


def test_adds_a_left_child():
    tree = BinaryTree()
    root = Node('R')
    tree.root = root
    left = Node('L')

    tree.addLeft(root, left)

    assert tree.getLeft(root) is left
    assert tree.getRight(root) is None


def test_stores_the_side_in_the_edge_properties():
    tree = BinaryTree()
    root = Node('R')
    tree.root = root

    tree.addLeft(root, Node('L'))

    edge = next(e for e in root.edges_out)

    assert edge.properties['side'] == 'left'


def test_raises_when_a_left_child_already_exists():
    tree = BinaryTree()
    root = Node('R')
    tree.root = root
    tree.addLeft(root, Node('L'))

    with pytest.raises(ValueError, match='Left child already exists'):
        tree.addLeft(root, Node('L2'))


def test_forwards_the_edge_kwargs():
    tree = BinaryTree()
    root = Node('R')
    tree.root = root

    tree.addLeft(root, Node('L'), weight=5, color='red', width=2, properties={'tag': 1})

    edge = next(e for e in root.edges_out)

    assert edge.weight == 5
    assert edge.color == 'red'
    assert edge.width == 2
    assert edge.properties == {'side': 'left', 'tag': 1}
