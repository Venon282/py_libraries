"""Tests for TreeNode.maxProductByOneSplitted."""

from py_libraries.graph.BinaryTreeNode import TreeNode


def test_returns_zero_for_an_empty_tree():
    assert TreeNode.maxProductByOneSplitted(None) == 0


def test_returns_zero_for_a_single_node():
    # A single node has no edge to split, so the product is 0.
    assert TreeNode.maxProductByOneSplitted(TreeNode(5)) == 0


def test_returns_the_maximum_product():
    # 1 + 2 + 3 = 6.
    # Splitting the edge 1-2 gives (6 - 2) * 2 = 8.
    # Splitting the edge 1-3 gives (6 - 3) * 3 = 9.
    root = TreeNode(1, left=TreeNode(2), right=TreeNode(3))

    assert TreeNode.maxProductByOneSplitted(root) == 9


def test_returns_zero_when_all_the_splits_are_negative():
    # 0 + (-1) + 1 = 0.
    # Splitting the edge 0-(-1) gives (0 - (-1)) * (-1) = -1.
    # Splitting the edge 0-1 gives (0 - 1) * 1 = -1.
    # Splitting at the root gives 0 * 0 = 0, which is the maximum.
    root = TreeNode(0, left=TreeNode(-1), right=TreeNode(1))

    assert TreeNode.maxProductByOneSplitted(root) == 0
