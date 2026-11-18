"""Tests for TreeNode.sum."""

from py_libraries.graph.BinaryTreeNode import TreeNode


def test_returns_zero_for_an_empty_tree():
    assert TreeNode.sum(None) == 0


def test_returns_the_value_for_a_single_node():
    assert TreeNode.sum(TreeNode(7)) == 7


def test_returns_the_sum_of_all_the_nodes():
    # 1 + 2 + 3 + 4 = 10
    root = TreeNode(1, left=TreeNode(2, left=TreeNode(4)), right=TreeNode(3))

    assert TreeNode.sum(root) == 10
