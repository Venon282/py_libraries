class Node:
    def __init__(self, edges=[], x=0, y=0, size=10, color='white', border_color='black',
                 border_width=1, name='Node', properties=[]):
        """
        Initializes a Node object with position, appearance, and additional properties.

        Parameters:
            x (float): The x-coordinate of the node.
            y (float): The y-coordinate of the node.
            size (int): The radius or size of the node.
            color (str): The fill color of the node.
            border_color (str): The color of the node's border.
            border_width (int): The width of the node's border.
            name (str): The name identifier for the node.
            properties (list): Additional properties for the node.
        """
        # Position
        self.x = x
        self.y = y

        # Appearance
        self.size = size
        self.color = color
        self.border_color = border_color
        self.border_width = border_width

        # Identification and additional properties
        self.name = name
        self.edges = edges
        self.properties = properties

    def move(self, dx, dy):
        """
        Moves the node by a given offset.

        Parameters:
            dx (float): The amount to move along the x-axis.
            dy (float): The amount to move along the y-axis.
        """
        self.x += dx
        self.y += dy

    def addProperty(self, prop):
        """
        Adds a new property to the node.

        Parameters:
            prop: The property to add.
        """
        self.properties.append(prop)

    def removeProperty(self, prop):
        """
        Removes a property from the node if it exists.

        Parameters:
            prop: The property to remove.
        """
        if prop in self.properties:
            self.properties.remove(prop)

    def toDict(self):
        """
        Returns a dictionary representation of the node.

        Returns:
            dict: A dictionary containing the node's data.
        """
        return {
            "name": self.name,
            "position": {"x": self.x, "y": self.y},
            "size": self.size,
            "color": self.color,
            "border_color": self.border_color,
            "border_width": self.border_width,
            "properties": self.properties
        }

    def __str__(self):
        """
        Returns a string representation of the node.

        Returns:
            str: A string summarizing the node.
        """
        return (f"{self.name}: Position ({self.x}, {self.y}), Size {self.size}, "
                f"Color {self.color}, Border {self.border_color} (width {self.border_width})")

    def draw(self, canvas):
        """
        Draws the node on a given canvas.

        This method assumes that the canvas object supports methods like
        create_oval and create_text (as in Tkinter).

        Parameters:
            canvas: The drawing surface or canvas.
        """
        # Calculate the bounding box for the oval representation of the node.
        x0 = self.x - self.size
        y0 = self.y - self.size
        x1 = self.x + self.size
        y1 = self.y + self.size

        # Draw the node as an oval.
        canvas.create_oval(x0, y0, x1, y1,
                           fill=self.color,
                           outline=self.border_color,
                           width=self.border_width)
        # Draw the node's name at its center.
        canvas.create_text(self.x, self.y, text=self.name)
