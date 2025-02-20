class LinkedList(Tree):
    def __init__(self, head=None):
        """
        Initializes a LinkedList. If a head node is provided, it is set as the root
        (and tail) of the list.
        
        Parameters:
            head (Node, optional): The first node of the linked list.
        """
        super().__init__(root=head)
        self.tail = head

    def append(self, new_node, weight=1, color='black', width=1, directed=True, name='Edge', properties=None):
        """
        Appends a new node at the end of the linked list.
        
        Parameters:
            new_node (Node): The node to be appended.
            weight (float): Weight for the connecting edge (default is 1).
            color (str): Color of the connecting edge (default is 'black').
            width (int): Line width of the connecting edge (default is 1).
            directed (bool): Whether the edge is directed (default is True).
            name (str): Label for the connecting edge (default is 'Edge').
            properties (list, optional): Additional properties for the edge.
        """
        if properties is None:
            properties = []
        
        # If the list is empty, new_node becomes the head and tail.
        if self.root is None:
            self.root = new_node
            self.tail = new_node
            self.addNode(new_node)
        else:
            # Ensure the tail pointer is up to date.
            # (This is extra precaution if the internal structure was modified.)
            current = self.root
            while True:
                next_nodes = [edge.end for edge in self.edges if edge.start == current]
                if not next_nodes:
                    break
                current = next_nodes[0]
            self.tail = current
            
            # Use the Tree's addChild to attach new_node as the child of the current tail.
            self.addChild(self.tail, new_node, weight=weight, color=color, width=width, 
                          directed=directed, name=name, properties=properties)
            self.tail = new_node

    def __str__(self):
        """
        Returns a string representation of the linked list in sequence form.
        
        Returns:
            str: A representation showing each node's name linked by arrows.
        """
        result = "LinkedList: "
        current = self.root
        while current:
            result += current.name + " -> "
            # In a linked list, each node should have at most one outgoing edge.
            next_node = None
            for edge in self.edges:
                if edge.start == current:
                    next_node = edge.end
                    break
            current = next_node
        result += "None"
        return result
