"""Tests for BinaryTree.addRight."""

import pytest
from py_libraries.graph.BinaryTree import BinaryTree
from py_libraries.graph.Node import Node


def test_adds_a_right_child():
    tree = BinaryTree()
    root = Node('R')
    tree.root = root
    right = Node('R2')

    tree.addRight(root, right)

    assert tree.getRight(root) is right
    assert tree.getLeft(root) is None


def test_stores_the_side_in_the_edge_properties():
    tree = BinaryTree()
    root = Node('R')
    tree.root = root

    tree.addRight(root, Node('R2'))

    edge = next(e for e in root.edges_out)

    assert edge.properties['side'] == 'right'


def test_raises_when_a_right_child_already_exists():
    tree = BinaryTree()
    root = Node('R')
    tree.root = root
    tree.addRight(root, Node('R2'))

    with pytest.raises(ValueError, match='Right child already exists'):
        tree.addRight(root, Node('R3'))


def test_forwards_the_edge_kwargs():
    tree = BinaryTree()
    root = Node('R')
    tree.root = root

    tree.addRight(
        root,
        Node('R2'),
        weight=3,
        color='blue',
        width=1,
        properties={'tag': 2},
    )

    edge = next(e for e in root.edges_out)

    assert edge.weight == 3
    assert edge.color == 'blue'
    assert edge.width == 1
    assert edge.properties == {'side': 'right', 'tag': 2}
