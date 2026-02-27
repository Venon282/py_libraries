from Graph import Graph
from Edge import Edge

class Tree(Graph):
    def __init__(self):
        """
        Initializes a Tree.
        
        Parameters:
            root (Node, optional): The root node of the tree. If provided, it is added to the tree.
        """
        super().__init__()
        self.root = None
    
    # def addChild(self, parent, child, weight=1, color='black', width=1, 
    #              directed=True, name='Edge', properties=None):
    #     """
    #     Adds a child node to a given parent in the tree.
        
    #     This method creates a directed edge from the parent to the child. It also
    #     checks that the parent is already in the tree and that adding the new edge
    #     will not introduce a cycle.
        
    #     Parameters:
    #         parent (Node): The parent node already in the tree.
    #         child (Node): The child node to add.
    #         weight (float): The weight associated with the edge (default is 1).
    #         color (str): The color of the edge (default is 'black').
    #         width (int): The width of the edge line (default is 1).
    #         directed (bool): Indicates if the edge is directed (default is True).
    #         name (str): A name or label for the edge (default is 'Edge').
    #         properties (list, optional): Additional properties for the edge.
        
    #     Raises:
    #         Exception: If the parent is not in the tree, the child is already in the tree,
    #                    or if adding the child would create a cycle.
    #     """
    #     if properties is None:
    #         properties = []
        
    #     if parent not in self.nodes:
    #         raise Exception("Parent node must already be part of the tree.")
        
    #     if child in self.nodes:
    #         raise Exception("Child node is already in the tree.")
        
    #     # Check for cycle: adding an edge from parent to child must not create one.
    #     if self._createsCycle(parent, child):
    #         raise Exception("Adding this child would create a cycle in the tree.")
        
    #     # Add the child and create the edge from parent to child.
    #     self.addNode(child)
    #     edge = Edge(start=parent, end=child, weight=weight, color=color, 
    #                 width=width, directed=directed, name=name, properties=properties)
    #     self.addEdge(edge)
    
    # def _createsCycle(self, parent, child):
    #     """
    #     Determines whether adding an edge from parent to child would create a cycle.
        
    #     In a tree, if there is already a path from the proposed child back to the parent,
    #     then adding the edge would form a cycle.
        
    #     Parameters:
    #         parent (Node): The prospective parent node.
    #         child (Node): The prospective child node.
        
    #     Returns:
    #         bool: True if a cycle would be created, False otherwise.
    #     """
    #     return self._hasPath(child, parent)
    
    # def _hasPath(self, current, target, visited=None):
    #     """
    #     Helper method that uses depth-first search (DFS) to determine if there is a path 
    #     from the current node to the target node.
        
    #     Parameters:
    #         current (Node): The node to start the search from.
    #         target (Node): The node to search for.
    #         visited (set): A set of already visited nodes to avoid loops.
        
    #     Returns:
    #         bool: True if a path exists, False otherwise.
    #     """
    #     if visited is None:
    #         visited = set()
    #     if current == target:
    #         return True
    #     visited.add(current)
    #     for edge in self.edges:
    #         # Only consider edges that go out from the current node.
    #         if edge.directed and edge.start == current:
    #             if edge.end not in visited:
    #                 if self._hasPath(edge.end, target, visited):
    #                     return True
    #     return False
    
    # def is_valid_tree(self):
    #     """
    #     Checks whether the current graph structure is a valid tree.
        
    #     A valid tree must:
    #       - Have a designated root.
    #       - Be connected (i.e. every node is reachable from the root).
    #       - Contain exactly n-1 edges if there are n nodes.
        
    #     Returns:
    #         bool: True if the structure is a valid tree, False otherwise.
    #     """
    #     if self.root is None:
    #         return False
    #     if len(self.edges) != len(self.nodes) - 1:
    #         return False
    #     visited = set()
    #     self._dfs(self.root, visited)
    #     return len(visited) == len(self.nodes)
    
    # def _dfs(self, node, visited):
    #     """
    #     Depth-first search helper method to mark all nodes reachable from the given node.
        
    #     Parameters:
    #         node (Node): The starting node for DFS.
    #         visited (set): A set to keep track of visited nodes.
    #     """
    #     visited.add(node)
    #     for edge in self.edges:
    #         if edge.directed and edge.start == node and edge.end not in visited:
    #             self._dfs(edge.end, visited)
    
    # def __str__(self):
    #     """
    #     Returns a string representation of the tree, including its root and
    #     the underlying graph information.
        
    #     Returns:
    #         str: A string summarizing the tree.
    #     """
    #     root_name = self.root.name if self.root else 'None'
    #     return f"Tree (root: {root_name})\n" + super().__str__()
