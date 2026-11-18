"""Tests for TreeNode.maxLevelSum."""

from py_libraries.graph.BinaryTreeNode import TreeNode


def test_returns_zero_for_an_empty_tree():
    assert TreeNode.maxLevelSum(None) == 0


def test_returns_one_for_a_single_node():
    assert TreeNode.maxLevelSum(TreeNode(5)) == 1


def test_returns_the_level_with_the_maximum_sum():
    # Level 1: 1, Level 2: 2 + 3 = 5, Level 3: 4
    root = TreeNode(1, left=TreeNode(2, left=TreeNode(4)), right=TreeNode(3))

    assert TreeNode.maxLevelSum(root) == 2


def test_returns_the_first_level_on_a_tie():
    # Level 1: 2, Level 2: 1 + 1 = 2. The first one is returned.
    root = TreeNode(2, left=TreeNode(1), right=TreeNode(1))

    assert TreeNode.maxLevelSum(root) == 1
