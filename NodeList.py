from Node import Node
from Edge import Edge


class ListNode(Node):
    def __init__(self, edges=None, x=0, y=0, size=10, color='white', border_color='black',
                 border_width=1, value='Node', properties=None, **kwargs):
        """
        Initializes a ListNode with inherited graphical and positional properties.
        The node's data is stored in 'value'. Its list links (previous/next) are represented
        by edges (max two: one incoming and one outgoing).
        
        Parameters:
            edges (list, optional): A list of Edge objects (default: empty list).
            x (float): x-coordinate.
            y (float): y-coordinate.
            size (int): Size of the node.
            color (str): Fill color.
            border_color (str): Border color.
            border_width (int): Border width.
            value (any): The node's data value.
            properties (list, optional): Additional properties.
            **kwargs: Any additional keyword arguments for Node.
        """
        if edges is None:
            edges = []
        if properties is None:
            properties = []
        kwargs.setdefault('edges', edges)
        kwargs.setdefault('x', x)
        kwargs.setdefault('y', y)
        kwargs.setdefault('size', size)
        kwargs.setdefault('color', color)
        kwargs.setdefault('border_color', border_color)
        kwargs.setdefault('border_width', border_width)
        kwargs.setdefault('value', value)
        kwargs.setdefault('properties', properties)
        super().__init__(**kwargs)
    
    def next(self):
        """
        Returns the next ListNode (i.e. the node connected by an outgoing edge),
        or None if no such edge exists.
        """
        for edge in self.edges:
            if edge.start == self:  # Outgoing edge
                return edge.end
        return None

    def setNext(self, next_node, weight=1, color='black', width=1, directed=True, edge_name='next', properties=None):
        """
        Sets the next pointer of this node by adding (or replacing) the outgoing edge.
        
        Parameters:
            next_node (ListNode): The node to be set as next.
            weight (float): Weight for the edge.
            color (str): Color for the edge.
            width (int): Edge width.
            directed (bool): Whether the edge is directed.
            edge_name (str): Name for the edge.
            properties (list, optional): Additional properties for the edge.
        """
        self.removeNext()  # Remove any existing outgoing (next) edge
        if properties is None:
            properties = []
        edge = Edge(start=self, end=next_node, weight=weight, color=color,
                    width=width, directed=directed, name=edge_name, properties=properties)
        # Add the edge to self and ensure it's also in the next_node's edges
        self.edges.append(edge)
        if edge not in next_node.edges:
            next_node.edges.append(edge)

    def removeNext(self):
        """
        Removes the outgoing edge representing the next pointer, if it exists.
        """
        to_remove = None
        for edge in self.edges:
            if edge.start == self:
                to_remove = edge
                break
        if to_remove:
            self.edges.remove(to_remove)
            if to_remove in to_remove.end.edges:
                to_remove.end.edges.remove(to_remove)

    def previous(self):
        """
        Returns the previous ListNode (i.e. the node connected by an incoming edge),
        or None if no such edge exists.
        """
        for edge in self.edges:
            if edge.end == self and edge.start != self:  # Incoming edge
                return edge.start
        return None

    def setPrevious(self, prev_node, weight=1, color='black', width=1, directed=True, edge_name='prev', properties=None):
        """
        Sets the previous pointer of this node by adding (or replacing) an incoming edge.
        
        Parameters:
            prev_node (ListNode): The node to be set as previous.
            weight (float): Weight for the edge.
            color (str): Color for the edge.
            width (int): Edge width.
            directed (bool): Whether the edge is directed.
            edge_name (str): Name for the edge.
            properties (list, optional): Additional properties for the edge.
        """
        self.removePrevious()
        if properties is None:
            properties = []
        edge = Edge(start=prev_node, end=self, weight=weight, color=color,
                    width=width, directed=directed, name=edge_name, properties=properties)
        # Add the edge to self and ensure it's also in the previous node's edges
        self.edges.append(edge)
        if edge not in prev_node.edges:
            prev_node.edges.append(edge)

    def removePrevious(self):
        """
        Removes the incoming edge representing the previous pointer, if it exists.
        """
        to_remove = None
        for edge in self.edges:
            if edge.end == self and edge.start != self:
                to_remove = edge
                break
        if to_remove:
            self.edges.remove(to_remove)
            if to_remove in to_remove.start.edges:
                to_remove.start.edges.remove(to_remove)

    def __str__(self):
        """
        Returns a string representation of the ListNode showing its value,
        along with its previous and next node values.
        """
        next_node = self.getNext()
        next_val = next_node.value if next_node is not None else "None"
        prev_node = self.getPrevious()
        prev_val = prev_node.value if prev_node is not None else "None"
        return f"ListNode(value={self.value}, prev={prev_val}, next={next_val})"
