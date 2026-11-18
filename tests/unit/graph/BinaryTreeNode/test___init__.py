"""Tests for TreeNode.__init__."""

from py_libraries.graph.BinaryTreeNode import TreeNode


def test_default_values():
    node = TreeNode()

    assert node.val == 0
    assert node.left is None
    assert node.right is None


def test_custom_values():
    left = TreeNode(1)
    right = TreeNode(2)
    node = TreeNode(3, left=left, right=right)

    assert node.val == 3
    assert node.left is left
    assert node.right is right


def test_children_are_independent():
    node = TreeNode(0)
    node.left = TreeNode(1)

    assert node.left is not None
    assert node.left.val == 1
    assert node.right is None
