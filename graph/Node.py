import random

class Node:
    
    def __init__(self, key='undefined', value=None,  properties={},
                 x=None, y=None, radius=2, 
                 color='white', edge_color='black', edge_width=1):
        # Identification and additional properties
        self.key = str(key)
        self.value = value
        self._edges = set()
        self._edges_in = set()
        self._edges_out = set()
        self.properties = properties
        
        # Position
        self.x = x if x is not None else 100*random.random()
        self.y = y if y is not None else 100*random.random()

        # Appearance
        self.radius = radius
        self.diameter = radius * 2
        self.color = color
        self.edge_color = edge_color
        self.edge_width = edge_width
        
    # region updates
    def addEdge(self, edge):
        is_start = edge.start_node is self
        is_end = edge.end_node is self
        
        if is_start:
            self._edges_out.add(edge)
            
        if is_end:
            self._edges_in.add(edge)
            
        if is_start or is_end:
            self._edges.add(edge)
            
    def removeEdge(self, edge):
        self._edges.remove(edge)
        self._edges_in.discard(edge)
        self._edges_out.discard(edge)
        
    def discardEdge(self, edge):
        self._edges.discard(edge)
        self._edges_in.discard(edge)
        self._edges_out.discard(edge)
        
    def move(self, dx, dy):
        """
        Moves the node by a given offset.

        Parameters:
            dx (float): The amount to move along the x-axis.
            dy (float): The amount to move along the y-axis.
        """
        self.x += dx
        self.y += dy
    
    # end region

    # region Properties
    @property
    def edges(self):
        return self._edges
    
    @property
    def edges_in(self):
        return self._edges_in
    
    @property
    def edges_out(self):
        return self._edges_out
    
    @property
    def degree_out(self):
        return len(self._edges_out)
    
    @property
    def degree_in(self):
        return len(self._edges_in)
    
    @property
    def coordinate(self):
        return (self.x, self.y)
    
    @property
    def nodesIn(self):
        return [edge.start for edge in self.edges_in]
    
    @property
    def nodesOut(self):
        return [edge.end for edge in self.edges_out]
    
    @property
    def neighbours(self):
        res = {e.end if e.start is self else e.start for e in self._edges}
        res.discard(self)
        return res
    
    # end region
    
    # region Cast
    def toDict(self):
        """
        Returns a dictionary representation of the node.

        Returns:
            dict: A dictionary containing the node's data.
        """
        return {
            "key": self.key,
            "value": self.value,
            "position": {"x": self.x, "y": self.y},
            "radius": self.radius,
            "diameter": self.diameter,
            "color": self.color,
            "edge_color": self.edge_color,
            "edge_width": self.edge_width,
            "properties": self.properties
        }
    # end region
    
    # region is
    def isSink(self):
        return self.degree_out == 0
    
    def isSource(self):
        return self.degree_in == 0
    
    # end region
    
    # region overwriting
    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return isinstance(other, Node) and self.key == other.key
    
    def __str__(self):
        """
        Returns a string representation of the node.

        Returns:
            str: A string summarizing the node.
        """
        return (f"{self.key} ({self.value}): Position ({self.x}, {self.y}), Diameter {self.diameter}, "
                f"Color {self.color}, Border {self.edge_color} (width {self.edge_width})")
    # end region
