import math

class Edge:
    def __init__(self, start, end, weight=1, color='black', width=1,
                 directed=False, name='Edge', properties=None):
        """
        Initializes an Edge object that connects two nodes.

        Parameters:
            start (Node): The starting node of the edge.
            end (Node): The ending node of the edge.
            weight (float): The weight or cost associated with the edge.
            color (str): The color used to draw the edge.
            width (int): The thickness of the edge line.
            directed (bool): Indicates if the edge is directed.
            name (str): A name or label for the edge.
            properties (list): Additional properties for the edge.
        """
        self.start = start
        self.end = end
        self.weight = weight
        self.color = color
        self.width = width
        self.directed = directed
        self.name = name
        self.properties = properties if properties is not None else []

    def length(self):
        """
        Calculates the Euclidean distance between the start and end nodes.

        Returns:
            float: The distance between the two nodes.
        """
        dx = self.start.x - self.end.x
        dy = self.start.y - self.end.y
        return math.sqrt(dx * dx + dy * dy)

    def addProperty(self, prop):
        """
        Adds a new property to the edge.

        Parameters:
            prop: The property to add.
        """
        self.properties.append(prop)

    def removeProperty(self, prop):
        """
        Removes a property from the edge if it exists.

        Parameters:
            prop: The property to remove.
        """
        if prop in self.properties:
            self.properties.remove(prop)

    def toDict(self):
        """
        Returns a dictionary representation of the edge.

        Returns:
            dict: A dictionary containing the edge's data.
        """
        return {
            "name": self.name,
            "start": self.start.name,
            "end": self.end.name,
            "weight": self.weight,
            "color": self.color,
            "width": self.width,
            "directed": self.directed,
            "properties": self.properties
        }

    def __str__(self):
        """
        Returns a string representation of the edge.

        Returns:
            str: A string summarizing the edge.
        """
        direction = "->" if self.directed else "--"
        return (f"{self.name}: {self.start.name} {direction} {self.end.name}, "
                f"Weight {self.weight}, Color {self.color}, Width {self.width}")

    def draw(self, canvas):
        """
        Draws the edge on a given canvas.

        This method assumes that the canvas object supports methods like
        create_line and create_text (as in Tkinter).

        Parameters:
            canvas: The drawing surface or canvas.
        """
        # Draw the line between the two nodes.
        if self.directed:
            # If directed, an arrow is drawn at the end.
            canvas.create_line(self.start.x, self.start.y, self.end.x, self.end.y,
                               fill=self.color, width=self.width, arrow='last')
        else:
            canvas.create_line(self.start.x, self.start.y, self.end.x, self.end.y,
                               fill=self.color, width=self.width)

        # Optionally, draw the edge's weight at the midpoint.
        mid_x = (self.start.x + self.end.x) / 2
        mid_y = (self.start.y + self.end.y) / 2
        canvas.create_text(mid_x, mid_y, text=str(self.weight))
