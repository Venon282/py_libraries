from collections import deque

# internal
from .Graph import Graph
from .Edge import Edge
from .Node import Node

class Tree(Graph):
    def __init__(self, name='undefined'):
        """
        Initializes a Tree.
        
        Parameters:
            root (Node, optional): The root node of the tree. If provided, it is added to the tree.
        """
        super().__init__(name=name, is_directed=True)
        self._root: Node | None = None
    
    # region construction
    def addChild(self, parent: Node, child: Node, weight=1, color='black', width=0.003, properties={}):
        """ 
        Adds a child node to a parent node in the tree.
        Creates the directed edge from parent to child.

        Parameters:
            parent   (Node): Parent node already present in the tree.
            child    (Node): Child node to be added.
            weight   (float): Edge weight (default 1).
            color    (str)  : Edge colour (default “black”).
            width    (float): Edge thickness (default 0.003).
            properties (dict): Additional properties.

        Raises:
            ValueError: If the parent is missing, if the child is already in the tree,
                        or if the addition would create a cycle.
        """
        if parent.key not in self.nodes:
            raise ValueError(f'The parent node "{parent.key}" is not in the tree.')

        if child.key in self.nodes:
            raise ValueError(f'The child node "{child.key}" is already in the tree.')
        
        if self.hasPathDirected(parent, child):
            raise ValueError(f'Add "{child.key}" as chil of "{parent.key}" would create a cycle.')
        
        edge = Edge(
            start=parent,
            end=child,
            weight=weight,
            color=color,
            width=width,
            properties=properties,
        )
        
        self.addEdge(edge)
        
    def addChildByKey(self, parent_key: str, child_key:str, **kwargs):
        """
        Shortcut: adds a child by passing the keys directly.
        Creates the child node if necessary.

        Parameters:
            parent_key (str): Key of the parent node.
            child_key  (str): Key of the child node.
        """
        parent = self.getNodeByKey(parent_key)
        
        if parent is None:
            raise ValueError(f'No node found with the key "{parent_key}"')
        
        child = self.getNodeByKey(child_key) or Node(child_key)
        self.addChild(parent, child, **kwargs)
    # end region
    
    # region is
    def isValidTree(self) -> bool:
        """
        Checks that the current structure is a valid tree:
          - A defined root (degree_in == 0, unique).
          - n-1 edges for n nodes.
          - All nodes are accessible from the root.

        Returns:
            bool: True if it is a valid tree.
        """
        if self.root is None:
            return False
        
        n = len(self.nodes)
        if n == 0:
            return False
        
        if len(self.edges) != n-1:
            return False
        
        visited = set()
        [_ for _ in self.dfsDirected(self.root, visited=visited)]
        return len(visited) == n
    
    def isParentOf(self, node_a: Node, node_b: Node) -> bool:
        return node_b is self.getParent(node_a)
    
    def isChildOf(self, node_a: Node, node_b: Node) -> bool:
        raise self.isParentOf(node_a=node_b, node_b=node_a)
    
    def isSiblingOf(self, node_a: Node, node_b: Node) -> bool:
        for sibling in self.getISiblings(node_a):
            if sibling is node_b:
                return True
        return False
    
    def isAnscestorOf(self, node_a: Node, node_b: Node) -> bool:
        for ancestor in self.getIAncestors(node_b):
            if ancestor is node_a:
                return True
        return False
    
    def isDescendantOf(self, node_a: Node, node_b: Node) -> bool:
        for descendant in self.getIDescendants(node_b):
            if descendant is node_a:
                return True
        return False
    
    def isLeaf(self, node: Node) -> bool:
        return len(node.edges_out) == 0
    
    # end region
    
    # region get
    def getParent(self, node: Node) -> Node | None:
        """ 
        Return the parent of a node
        
        Returns:
            Node | None
        """
        for edge in node.edges_in:
            return edge.start
        
        return None

    def getChildrens(self, node: Node) -> list[Node]:
        """ 
        Return the childrens of a node
        
        Returns:
            list[Node]
        """
        return [edge.end for edge in node.edges_out]
    
    def getIChildrens(self, node: Node):
        """ 
        Return the childrens of a node

        """
        for edge in node.edges_out:
            yield edge
            
    def getSiblings(self, node: Node) -> list[Node]:
        """
        Return siblings nodes (same nodes except himself).

        Returns:
            list[Node]
        """
        parent = self.getParent(node)
        if parent is None:
            return []
        return [child for child in self.getIChildrens(parent) if child is not node]
    
    def getISiblings(self, node: Node):
        """
        Return siblings nodes (same nodes except himself).
        """
        parent = self.getParent(node)
        if parent is None:
            return []
        
        for child in self.getIChildrens(parent):
            if child is node:
                continue
            
            yield child
    
    def getDepth(self, node: Node) -> int:
        """ 
        Returns the depth of a node (distance from the root).
        The root is at depth 0.

        Returns:
            int: The depth of the node.

        Raises:
            ValueError: If the root is not defined.
        """
        if self.root is None:
            raise ValueError('The root is not defined')
        
        depth = 0
        current = node
        while current is not self.root:
            parent = self.getParent(node)
            if parent is None:
                raise ValueError('The node is not connected to the root.')
            
            current = parent
            depth +=1
            
        return depth
    
    def getHeight(self, node:Node = None) -> int:
        """ 
        Returns the height of the tree (or of the subtree rooted at 'node').
        A leaf node has a height of 0.

        Parameters:
            node (Node): Root of the subtree (default: root of the tree).

        Returns:
            int: Height.
        """
        node = node or self.root
        if node is None:
            return -1
        
        
        que = deque([(node, 0)])
        max_height = 0
        
        while que:
            node, height = que.popleft()
            
            max_height = max(max_height, height)
            
            for children in self.getIChildrens(node):
                que.append((children, height+1))
        
        return max_height
    
    def getLevel(self, depth: int) -> list[Node]:
        """
        Returns all nodes at a given depth.

        Parameters:
            depth (int): Target level (0 = root).

        Returns:
            list[Node]
        """
        if self.root is None:
            return []

        result = []
        queue = deque([(self.root, 0)])

        while queue:
            node, d = queue.popleft()
            if d == depth:
                result.append(node)
            elif d < depth:
                for children in self.getIChildrens(node):
                    queue.append((children, d + 1))

        return result
    
    def getLeaves(self) -> list[Node]:
        """
        Return all tree leaf (without childs)

        Returns:
            list[Node]
        """
        return [node for node in self.nodes.values() if node.degree_out == 0]
    
    def getAncestors(self, node: Node) -> list[Node]:
        """ 
        Return the ancestors ordonate node list (from parents to root)
        
        Returns:
            list[Node]
        """
        ancestors = []
        current = self.getParent(node)
        while current is not None:
            ancestors.append(current)
            current = self.getParent(current)
        return ancestors
    
    def getIAncestors(self, node: Node):
        """ 
        Return the ancestors ordonate node list (from parents to root)

        """
        current = self.getParent(node)
        while current is not None:
            yield current
            current = self.getParent(current)
    
    def getDescendants(self, node: Node) -> list[Node]:
        """ 
        Return the descendants ordonate node list (from parents to root)
        
        Returns:
            list[Node]
        """
        descendants = []
        queue = deque(self.getChildrens(node))
        visited = set()
   
        while queue:
            current = queue.popleft()
            if current in visited:
                continue

            visited.add(current)
            descendants.append(current)
            queue.extend(self.getChildrens(current))
            
        return descendants
            
    def getIDescendants(self, node: Node):
        """ 
        Return the descendants ordonate node list (from parents to root)
        """
        queue = deque(self.getChildrens(node))
        visited = set()
   
        while queue:
            current = queue.popleft()
            if current in visited:
                continue

            visited.add(current)
            yield current
            queue.extend(self.getChildrens(current))
            
    def getLowestCommonAncestor(self, node_a: Node, node_b: Node) -> Node | None:
        """
        Lowest Common Ancestor (LCA) - ancêtre the closest ancestor.

        Returns:
            Node | None:Common ancestor, or None if inexistant.
        """
        node_a_ancestors = set(self.getAncestors(node_a))
        
        for ancestor in self.getIAncestors(node_b):
            if ancestor in node_a_ancestors:
                return ancestor
        return None
    # end region
    
    
    # region properties
    @property
    def root(self):
        return self._root
    # end region
    
    # region setter
    @root.setter
    def root(self, node: Node):
        """ 
        Sets the root of the tree.
        If the node is not yet in the graph, it is added.

        Parameters:
            node (Node): The root node.

        Raises:
            ValueError: If a node with an in_degree > 0 is set as the root.
        """
        if node.degree_in > 0:
            raise ValueError(f'The node "{node.key}" have parents so it can\'t be a root.')
        
        self.addNode(node)
        self._root = node
    # end region
    
    # region cast
    def toDict(self) -> dict:
        """
        Extend the dict representation of Graph with the root key

        Returns:
            dict
        """
        base = super().toDict()
        base['root'] = self.root.key if self.root else None
        return base
    # end region
    
    # region overwriting
    def __str__(self) -> str:
        root_key = self.root.key if self.root else 'None'
        n = len(self.nodes)
        h = self.getHeight() if self.root else '?'
        leaves = len(self.getLeaves())
        return (
            f"--- Tree '{self.name}' ---\n"
            f"Root    : {root_key}\n"
            f"Nodes   : {n}  |  Edges : {len(self.edges)}\n"
            f"Height  : {h}  |  Leaves: {leaves}\n"
            f"Valid   : {self.isValidTree()}"
        )
    # end region
    
    