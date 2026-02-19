import math
import numpy as np

class Edge:
    def __init__(self, start, end, weight=1, properties={},
                 color='black', width=0.003):
        self.start = start
        self.end = end
        self.weight = weight
        self.properties = properties
        
        # aspect
        self.color = color
        self.width = width
    
    # region Properties
    @property
    def dx(self):
        return self.end.x - self.start.x
    
    @property
    def dy(self):
        return self.end.y - self.start.y
    
    @property
    def length(self):
        return np.hypot(self.dx, self.dy)
    
    @property
    def coordinate(self):
        dx = self.dx
        dy = self.dy
        
        length = np.hypot(dx, dy)
        
        x_normalisation = dx / length
        y_normalisation = dy / length
        
        start_x = self.start.x + self.start.radius * x_normalisation
        start_y = self.start.y + self.start.radius * y_normalisation
        
        end_x = self.end.x - self.end.radius * x_normalisation
        end_y = self.end.y - self.end.radius * y_normalisation
        
        return (start_x, start_y), (end_x, end_y)
    # endregion
    
    # region Cast
    def toDict(self):
        """
        Returns a dictionary representation of the edge.

        Returns:
            dict: A dictionary containing the edge's data.
        """
        return {
            "start": self.start.key,
            "end": self.end.key,
            "weight": self.weight,
            "properties": self.properties
         }
    # end region
    
    # region overwriting
    def __hash__(self):
        # For undirected graph, make start/end order-independent
        # if not self.start.graph.is_directed:
        #     return hash(frozenset([self.start, self.end]))
        return hash((self.start, self.end))

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        # if not self.start.graph.is_directed:
        #     return frozenset([self.start, self.end]) == frozenset([other.start, other.end])
        return self.start == other.start and self.end == other.end
    
    def __str__(self):
        """
        Returns a string representation of the edge.

        Returns:
            str: A string summarizing the edge.
        """
        return (f"{self.start.name} -- {self.end.name}, "
                f"Weight {self.weight}, Color {self.color}, Width {self.width}")
    # end region
        