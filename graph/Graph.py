class Graph:
    def __init__(self):
        """
        Initializes an empty Graph with lists for nodes and edges.
        """
        self.nodes = []
        self.edges = []

    def addNode(self, node):
        """
        Adds a node to the graph.

        Parameters:
            node (Node): The node instance to add.
        """
        if node not in self.nodes:
            self.nodes.append(node)

    def removeNode(self, node):
        """
        Removes a node from the graph, along with any edges connected to it.

        Parameters:
            node (Node): The node instance to remove.
        """
        try:
            self.nodes.remove(node)
            # Remove all edges that are connected to this node.
            self.edges = [edge for edge in self.edges if edge.start != node and edge.end != node]
            # remove the edge from the node's own edge list.
            node.edges = []
        except:
            pass
            
    def addEdge(self, edge):
        """
        Adds an edge to the graph and also registers the edge with the corresponding nodes.

        Parameters:
            edge (Edge): The edge instance to add.
        """
        # Ensure both nodes are part of the graph.
        if edge.start not in self.nodes:
            self.addNode(edge.start)
        if edge.end not in self.nodes:
            self.addNode(edge.end)
        self.edges.append(edge)
        
        # Add this edge to the nodes' internal edge lists.
        if edge not in edge.start.edges:
            edge.start.edges.append(edge)
        if edge not in edge.end.edges:
            edge.end.edges.append(edge)

    def removeEdge(self, edge):
        """
        Removes an edge from the graph.

        Parameters:
            edge (Edge): The edge instance to remove.
        """
        try:
            self.edges.remove(edge)
        except:
            pass
        try:
            edge.start.edges.remove(edge)
        except:
            pass
        try:
            edge.end.edges.remove(edge)
        except:
            pass

    def getNode(self, name):
        """
        Retrieves a node by its name.

        Parameters:
            name (str): The name identifier of the node.

        Returns:
            Node or None: The node with the matching name, or None if not found.
        """
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def isLinkedList(self):
        def forward(node):
            if len(node.edges) > 2 or len(node.edges) == 0:
                return False
            if len(node.edges) == 1 and node.edges[0].end is node:
                return True
            for edge in node.edges:
                if edge.start is node:
                    return forward(edge.end)
            return False

        def backward(node):
            if len(node.edges) > 2 or len(node.edges) == 0:
                return False
            if len(node.edges) == 1 and node.edges[0].start is node:
                return True
            for edge in node.edges:
                if edge.end is node:
                    return forward(edge.start)
            return False

        if len(self.nodes) == 0 and len(self.nodes[0].edges) == 0:
            return False

        node = self.nodes[0]
        return forward(node) and backward(node)



    def toDict(self):
        """
        Returns a dictionary representation of the graph.

        Returns:
            dict: A dictionary with lists of nodes and edges.
        """
        return {
            "nodes": [node.toDict() for node in self.nodes],
            "edges": [edge.toDict() for edge in self.edges]
        }

    def __str__(self):
        """
        Returns a string representation of the graph.

        Returns:
            str: A summary of the graph's nodes and edges.
        """
        nodes_str = ', '.join([node.name for node in self.nodes])
        edges_str = ', '.join([
            f"{edge.start.name}->{edge.end.name}" if edge.directed else f"{edge.start.name}-{edge.end.name}"
            for edge in self.edges
        ])
        return f"Graph with nodes: {nodes_str}\nEdges: {edges_str}"

    def draw(self, canvas):
        """
        Draws the entire graph on a given canvas.

        Parameters:
            canvas: The drawing surface or canvas (e.g., a Tkinter Canvas).
        """
        # Draw all edges first so that nodes appear on top.
        for edge in self.edges:
            edge.draw(canvas)
        # Then draw all nodes.
        for node in self.nodes:
            node.draw(canvas)
