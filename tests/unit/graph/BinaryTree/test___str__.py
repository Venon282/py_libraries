"""Tests for BinaryTree.__str__."""

from py_libraries.graph.BinaryTree import BinaryTree
from py_libraries.graph.Node import Node


def test_str_contains_the_summary(binary_tree):
    text = str(binary_tree)

    assert 'BinaryTree (root: R)' in text
    assert "--- Tree 'undefined' ---" in text
    assert 'Root    : R' in text
    assert 'Nodes   : 3  |  Edges : 2' in text
    assert 'Height  : 1  |  Leaves: 2' in text
    assert 'Valid   : True' in text


def test_str_shows_none_when_there_is_no_root():
    text = str(BinaryTree())

    assert 'BinaryTree (root: None)' in text
    assert 'Root    : None' in text
    assert 'Valid   : False' in text


def test_str_uses_the_name():
    tree = BinaryTree('bt')
    tree.root = Node('R')

    assert "--- Tree 'bt' ---" in str(tree)
