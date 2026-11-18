"""A standalone binary tree node with level-based static utilities."""

from __future__ import annotations

from collections import deque


class TreeNode:
    """A node of a standalone binary tree.

    Unlike the graph-based `BinaryTree`, this node is self-contained: it
    carries a value and direct references to its left and right children, and
    exposes static utilities that operate on a whole tree.

    Args:
        val: Value carried by the node.
        left: Left child of the node.
        right: Right child of the node.
    """

    def __init__(
        self,
        val: int = 0,
        left: TreeNode | None = None,
        right: TreeNode | None = None,
    ) -> None:
        self.val: int = val
        self.left: TreeNode | None = left
        self.right: TreeNode | None = right

    @staticmethod
    def maxLevelSum(root: TreeNode | None) -> int:
        """Return the level with the maximum sum of node values.

        The levels are counted from 1, starting at the root. When several
        levels share the same maximum sum, the first one is returned.

        Args:
            root: Root of the tree.

        Returns:
            The 1-based level with the maximum sum, or 0 if the tree is empty.
        """
        if root is None:
            return 0

        max_sum = float('-inf')
        ans = 0
        level = 0

        q: deque[TreeNode] = deque()
        q.append(root)

        while q:
            level += 1
            sum_at_current_level = 0
            # Iterate over all the nodes in the current level.
            for _ in range(len(q)):
                node = q.popleft()
                sum_at_current_level += node.val

                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)

            if max_sum < sum_at_current_level:
                max_sum, ans = sum_at_current_level, level

        return ans

    @staticmethod
    def sum(root: TreeNode | None) -> int:
        """Return the sum of the values of all the nodes of the tree.

        Args:
            root: Root of the tree.

        Returns:
            The sum of the node values, or 0 if the tree is empty.
        """
        if root is None:
            return 0

        total = 0
        q: deque[TreeNode] = deque()
        q.append(root)
        while q:
            for _ in range(len(q)):
                node = q.popleft()
                total += node.val
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
        return total

    @staticmethod
    def maxProductByOneSplitted(root: TreeNode | None) -> int:
        """Return the maximum product obtained by splitting the tree once.

        The tree is split by removing one edge, which separates it into two
        subtrees. The product of the sum of the two subtrees is maximized
        over all the possible splits.

        Args:
            root: Root of the tree.

        Returns:
            The maximum product of the sums of the two subtrees, or 0 if the
            tree is empty.
        """
        if root is None:
            return 0

        ans = float('-inf')
        total = 0

        def dfs(node: TreeNode | None) -> int:
            nonlocal ans, total
            if node is None:
                return 0
            cur_sum = node.val + dfs(node.left) + dfs(node.right)
            ans = max(ans, (total - cur_sum) * cur_sum)
            return cur_sum

        total = TreeNode.sum(root)
        dfs(root)
        return int(ans)
