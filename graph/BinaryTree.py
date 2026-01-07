from Tree import Tree

class BinaryTree(Tree):
    def __init__(self, root=None):
        """
        Initializes a BinaryTree.

        Parameters:
            root (Node, optional): The root node of the binary tree.
                                   If provided, it is added to the tree.
        """
        super().__init__(root)

    def addChild(self, parent, child, weight=1, color='black', width=1,
                 directed=True, name='Edge', properties=None):
        """
        Override the generic addChild method.
        For a BinaryTree, please use addLeft or addRight to ensure the binary constraints.
        """
        raise Exception("For BinaryTree, please use addLeft or addRight.")

    def addLeft(self, parent, child, weight=1, color='black', width=1,
                directed=True, name='LeftEdge', properties=None):
        """
        Adds a left child to the specified parent node.

        Parameters:
            parent (Node): The parent node already in the tree.
            child (Node): The left child node to add.
            weight (float): The weight of the edge (default is 1).
            color (str): The color of the edge (default is 'black').
            width (int): The width of the edge line (default is 1).
            directed (bool): Should be True since the edge points from parent to child.
            name (str): The name or label for the edge (default is 'LeftEdge').
            properties (list, optional): Additional properties for the edge.
                                         The property "left" will be added automatically.

        Returns:
            The created edge (by calling the parent's addChild method).
        """
        if properties is None:
            properties = []
        if "left" not in properties:
            properties.append("left")
        # Ensure that the parent does not already have a left child.
        if self.getLeft(parent) is not None:
            raise Exception("Left child already exists for this parent.")
        # Use Tree's addChild (which does cycle checking, node addition, etc.)
        return super().addChild(parent, child, weight=weight, color=color,
                                  width=width, directed=directed, name=name,
                                  properties=properties)

    def addRight(self, parent, child, weight=1, color='black', width=1,
                 directed=True, name='RightEdge', properties=None):
        """
        Adds a right child to the specified parent node.

        Parameters:
            parent (Node): The parent node already in the tree.
            child (Node): The right child node to add.
            weight (float): The weight of the edge (default is 1).
            color (str): The color of the edge (default is 'black').
            width (int): The width of the edge line (default is 1).
            directed (bool): Should be True since the edge points from parent to child.
            name (str): The name or label for the edge (default is 'RightEdge').
            properties (list, optional): Additional properties for the edge.
                                         The property "right" will be added automatically.

        Returns:
            The created edge (by calling the parent's addChild method).
        """
        if properties is None:
            properties = []
        if "right" not in properties:
            properties.append("right")
        # Ensure that the parent does not already have a right child.
        if self.getRight(parent) is not None:
            raise Exception("Right child already exists for this parent.")
        # Use Tree's addChild method.
        return super().addChild(parent, child, weight=weight, color=color,
                                  width=width, directed=directed, name=name,
                                  properties=properties)

    def getLeft(self, parent):
        """
        Retrieves the left child of a given parent node.

        Parameters:
            parent (Node): The parent node.

        Returns:
            Node or None: The left child node if it exists; otherwise, None.
        """
        for edge in self.edges:
            if edge.directed and edge.start == parent and "left" in edge.properties:
                return edge.end
        return None

    def getRight(self, parent):
        """
        Retrieves the right child of a given parent node.

        Parameters:
            parent (Node): The parent node.

        Returns:
            Node or None: The right child node if it exists; otherwise, None.
        """
        for edge in self.edges:
            if edge.directed and edge.start == parent and "right" in edge.properties:
                return edge.end
        return None

    def getChildren(self, parent):
        """
        Returns a list of children for a given parent node in left-right order.

        Parameters:
            parent (Node): The parent node.

        Returns:
            list: A list containing the left child (if any) followed by the right child (if any).
        """
        children = []
        left = self.getLeft(parent)
        right = self.getRight(parent)
        if left is not None:
            children.append(left)
        if right is not None:
            children.append(right)
        return children

    # def isUnivalTree(root):
    #     """
    #     :type root: Optional[TreeNode]
    #     :rtype: bool
    # A binary tree is uni-valued if every node in the tree has the same value.

    # Given the root of a binary tree, return true if the given tree is uni-valued, or false otherwise.

    

    # Example 1:


    # Input: root = [1,1,1,1,1,null,1]
    # Output: true
    # Example 2:


    # Input: root = [2,2,2,5,2]
    # Output: false
    #     """
    #     cond = True
    #     if root.left:
    #         cond = cond and root.val == root.left.val and self.isUnivalTree(root.left)
    #     if root.right:
    #         cond = cond and root.val == root.right.val and self.isUnivalTree(root.right)
    #     return cond

    def __str__(self):
        """
        Returns a string representation of the binary tree, including the root and
        underlying graph information.

        Returns:
            str: A summary of the binary tree.
        """
        root_name = self.root.name if self.root else 'None'
        return f"BinaryTree (root: {root_name})\n" + super().__str__()
