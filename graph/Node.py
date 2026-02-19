import random

class Node:
    
    def __init__(self, key='undefined', value=None,  properties={},
                 x=None, y=None, radius=2, 
                 color='white', edge_color='black', edge_width=1):
        # Identification and additional properties
        self.key = str(key)
        self.value = value
        self.edges = set()
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
    
    # end region
        
    # region Properties
    @property
    def coordinate(self):
        return (self.x, self.y)
    # end region
    def move(self, dx, dy):
        """
        Moves the node by a given offset.

        Parameters:
            dx (float): The amount to move along the x-axis.
            dy (float): The amount to move along the y-axis.
        """
        self.x += dx
        self.y += dy
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
