from Stack import Stack
from Queue import Queue
from Graph import Graph
from Edge import Edge
from number import isPowerOfTwo

# todo remake it Graph->Tree->BinaryTree
# todo BinaryTree must be graph and not nodes
class BinaryTree(Graph):
    def __init__(self, val=None, left=None, right=None, is_root=False):
        super().__init__()
        
        self.val = val
        self.left = left
        self.right = right
        self.height = 0
        self.size = 1
        self.is_root = is_root
        self.nb_leaf = 0
        
        if self.left is not None:
            self.size += self.left.size
            self.nb_leaf += self.left.nb_leaf
            self.addEdge(Edge(self, self.left))
            
        if self.right is not None:
            self.size += self.right.size
            self.nb_leaf += self.right.nb_leaf
            self.addEdge(Edge(self, self.right))
            
        if self.nb_leaf == 0:
            self.nb_leaf = 1
            
        self.addNode(self)

    def build(self, lst):
        def buildRec(index, height=0):
            # If the index is out of bounds, return None (no node)
            if index >= len(lst) or lst[index] is None:
                return None, height -1

            # Create the node at the current index
            node = BinaryTree(lst[index])


            # Recursively build the left and right subtrees
            node.left, height_left = buildRec(2 * index + 1, height)
            node.right, height_right = buildRec(2 * index + 2, height)
            
            if node.left:
                node.addEdge(Edge(node, node.left))
            if node.right:
                node.addEdge(Edge(node, node.right))

            # Set the node's height to the maximum of its own height and the height of its children
            node.height = max(height, height_left, height_right)
            node.size = (node.left.size if node.left is not None else 0) + (node.right.size if node.right is not None else 0) + 1
            if node.left is None and node.right is None:
                node.nb_leaf = 1
            else:
                node.nb_leaf = (node.left.nb_leaf if node.left is not None else 0) + (node.right.nb_leaf if node.right is not None else 0)

            print(node.__dict__)
            return node, node.height + 1

        # Start building the tree from the root (index 0)
        tree, _ = buildRec(0, self.height)

        self.__dict__ = tree.__dict__
        self.is_root = True

    def updateSizes(self):
        def updateSizesRec(node):
            if node is None:
                return 0
            size = 1 + updateSizesRec(node.left) + updateSizesRec(node.right)
            node.size = size
            return size
        return updateSizesRec(self)

    def updateIsRoot(self, is_root=True):
        def updateIsRootRec(node, is_root):
            if nore is None:
                return
            node.is_root = is_root
            updateIsRootRec(node.left, False)
            updateIsRootRec(node.right, False)
        updateIsRootRec(self, is_root)

    def updateNbLeaf(self):
        def updateNbLeafRec(node):
            if node is None:
                return 0
            if node.left is None and node.right is None:
                node.nb_leaf = 1
                return node.nb_leaf
            node.nb_leaf = updateNbLeafRec(node.left) + updateNbLeafRec(node.right)



    def isPerfect(self):
        return self.size == 2**(self.height -1)

    def isFull(self):
        return self.size >= 2 * self.height + 1

    def isRoot(self):
        return self.is_root

    def isLeaf(self):
        return self.left is None and self.right is None

    def isComplete(self):
        queue = Queue([self])
        found_none = False  # Flag to check if we've encountered a None node

        while queue:
            node = queue.dequeue()

            # If we encounter a None node, mark the flag
            if node is None:
                found_none = True
            else:
                # If we've previously encountered a None node and now a valid node appears, it's not complete
                if found_none:
                    return False

                # Add the left and right children to the queue
                queue.enqueue(node.left)
                queue.enqueue(node.right)

        return True

    def isBalanced(self):
        def isBalancedRec(node):
            # Base case: If the node is None, it's balanced and has height -1
            if node is None:
                return True, -1  # balanced, height is -1

            # Check the balance of the left subtree
            left_balanced, left_height = isBalancedRec(node.left)
            if not left_balanced:
                return False, 0  # Unbalanced subtree, no need to continue

            # Check the balance of the right subtree
            right_balanced, right_height = isBalancedRec(node.right)
            if not right_balanced:
                return False, 0  # Unbalanced subtree, no need to continue

            # Check if the current node is balanced
            if abs(left_height - right_height) > 1:
                return False, 0  # Unbalanced at this node

            # Return True, and the current node's height
            return True, 1 + max(left_height, right_height)

        # Call check_balance starting from the root
        balanced, _ = isBalancedRec(self)
        return balanced

    def isDegenerate(self):
        def isDegenerateRec(node):
            if node is None:
                return True

            if node.left is not None and node.right is not None:
                return False
            return isDegenerateRec(node.left) and isDegenerateRec(node.right)
        
    def isSame(self, tree):
        def isSameRec(t1, t2):
            if t1 is None and t2 is None:
                return True
            if t1 is not None and t2 is not None and t1.val == t2.val:
                return isSameRec(t1.left, t2.left) and isSameRec(t1.right, t2.right)
            return False
        return isSameRec(self, tree)

    def toListMiddle(self):
        # Helper function to perform in-order traversal and collect values in a list
        def toListMiddleRec(node):
            if node is None:
                return []
            # Traverse the left subtree, then root, then right subtree
            return toListMiddleRec(node.left) + [node.val] + toListMiddleRec(node.right)

        # Start the in-order traversal from the root and return the list of values
        return toListMiddleRec(self.root)

    def toList(self):
        result = []
        # Helper function to fill the list based on the tree structure
        def toListRec(node, index):
            if node is None:
                return
            # Ensure the result list has enough elements to hold the index
            while len(result) <= index:
                result.append(None)

            # Set the value at the current index
            result[index] = node.val

            # Recurse for left and right children
            toListRec(node.left, 2 * index + 1)
            toListRec(node.right, 2 * index + 2)

        # Start filling the list starting from the root (index 0)
        toListRec(self, 0)
        return result


    def nbChilds(self):
        return sum([self.left is not None, self.right is not None])

    def display(self):
        """
        Recursively prints the binary tree sideways.

        Args:
            level (int): The current level in the tree (used for indentation).
            prefix (str): A label to show before the node's value.
        """
        def displayRec(node, level=0, prefix="Root: "):
            # First, recursively display the right subtree, increasing the level (indentation).
            if node.right:
                displayRec(node.right, level=level + 1, prefix="R--- ")

            # Print the current node with indentation.
            print("       " * level + prefix + str(node.height))

            # Then, recursively display the left subtree.
            if node.left:
                displayRec(node.left, level=level + 1, prefix="L--- ")
        displayRec(self)

if '__main__' == __name__:
    tree = BinaryTree()
    tree.build([1,2,3,4,5,None,8,None,None,6,7,None, None,9])
    print(tree.__dict__)
    tree.display()
    print(tree.toList())
    print('is completed: ',tree.isComplete())
    print('is balanced: ',tree.isBalanced())
