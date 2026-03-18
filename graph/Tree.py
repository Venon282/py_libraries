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
    
    # end region
    
    # region get
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
            if len(node.edges_in) == 0:
                raise ValueError('The node is not connected to the root.')
            
            current = node.edges_in[0]
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
            
            for edge in node.edges_out:
                que.append((edge.end, height+1))
        
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
                for edge in node.edges_out:
                    queue.append((edge.end, d + 1))

        return result
    
    def getLeaves(self) -> list[Node]:
        """
        Return all tree leaf (without childs)

        Returns:
            list[Node]
        """
        return [node for node in self.nodes.values() if node.degree_out == 0]

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
    
    