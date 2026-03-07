import collections

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    @staticmethod
    def maxLevelSum(root: TreeNode) -> int:
        max_sum, ans, level = float('-inf'), 0, 0

        q = collections.deque()
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
    def sum(root: TreeNode) -> int:
        somme = 0
        q = collections.deque()
        q.append(root)
        while q:
            for _ in range(len(q)):
                node = q.popleft()
                somme += node.val
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
        return somme
    
    @staticmethod
    def maxProductByOneSplitted(root: TreeNode) -> int:
        ans, total=float('-inf'), 0
        def dfs(root):
            nonlocal ans, total
            if not root: return 0
            cur_sum = root.val + dfs(root.left) + dfs(root.right)
            ans = max(ans, (total - cur_sum) * cur_sum)
            return cur_sum
        total = TreeNode.sum(root)
        dfs(root)
        return ans 