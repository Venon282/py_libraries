"""A node of a graph: an identified value with a 2D position and appearance."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .Edge import Edge

DEFAULT_RADIUS: float = 2.0
DEFAULT_EDGE_WIDTH: float = 1.0
DEFAULT_COLOR: str = 'white'
DEFAULT_EDGE_COLOR: str = 'black'


class Node:
    """A node of a graph.

    A node is identified by a string key, carries an optional value and a
    dictionary of additional properties, and has a 2D position and an
    appearance used by the drawing methods.

    Args:
        key: Key of the node. It is converted to a string.
        value: Value carried by the node.
        properties: Additional properties of the node. A new empty dictionary
            is used when `None`.
        x: X coordinate of the node. A random coordinate in `[0, 100)` is
            used when `None`.
        y: Y coordinate of the node. A random coordinate in `[0, 100)` is
            used when `None`.
        radius: Radius of the node as drawn.
        color: Fill color of the node as drawn.
        edge_color: Border color of the node as drawn.
        edge_width: Border width of the node as drawn.
    """

    def __init__(
        self,
        key: str = 'undefined',
        value: object = None,
        properties: dict[str, object] | None = None,
        x: float | None = None,
        y: float | None = None,
        radius: float = DEFAULT_RADIUS,
        color: str = DEFAULT_COLOR,
        edge_color: str = DEFAULT_EDGE_COLOR,
        edge_width: float = DEFAULT_EDGE_WIDTH,
    ) -> None:
        self.key: str = str(key)
        self.value: object = value
        self.properties: dict[str, object] = (
            properties if properties is not None else {}
        )
        self._edges: set[Edge] = set()
        self._edges_in: set[Edge] = set()
        self._edges_out: set[Edge] = set()
        self.x: float = x if x is not None else 100 * random.random()
        self.y: float = y if y is not None else 100 * random.random()
        self.radius: float = radius
        self.diameter: float = radius * 2
        self.color: str = color
        self.edge_color: str = edge_color
        self.edge_width: float = edge_width

    # region updates
    def addEdge(self, edge: Edge) -> None:
        """Register an edge that starts or ends at this node.

        Args:
            edge: Edge to register. It must have this node as start or end.
        """
        is_start = edge.start is self
        is_end = edge.end is self

        if is_start:
            self._edges_out.add(edge)

        if is_end:
            self._edges_in.add(edge)

        if is_start or is_end:
            self._edges.add(edge)

    def removeEdge(self, edge: Edge) -> None:
        """Unregister an edge from this node.

        Args:
            edge: Edge to unregister.

        Raises:
            KeyError: If the edge is not registered on this node.
        """
        self._edges.remove(edge)
        self._edges_in.discard(edge)
        self._edges_out.discard(edge)

    def discardEdge(self, edge: Edge) -> None:
        """Unregister an edge from this node if it is present.

        Args:
            edge: Edge to unregister.
        """
        self._edges.discard(edge)
        self._edges_in.discard(edge)
        self._edges_out.discard(edge)

    def move(self, dx: float, dy: float) -> None:
        """Move the node by a given offset.

        Args:
            dx: Offset along the x axis.
            dy: Offset along the y axis.
        """
        self.x += dx
        self.y += dy

    # endregion

    # region properties
    @property
    def edges(self) -> set[Edge]:
        """Edges that start or end at this node."""
        return self._edges

    @property
    def edges_in(self) -> set[Edge]:
        """Edges that end at this node."""
        return self._edges_in

    @property
    def edges_out(self) -> set[Edge]:
        """Edges that start at this node."""
        return self._edges_out

    @property
    def degree_out(self) -> int:
        """Number of edges that start at this node."""
        return len(self._edges_out)

    @property
    def degree_in(self) -> int:
        """Number of edges that end at this node."""
        return len(self._edges_in)

    @property
    def coordinate(self) -> tuple[float, float]:
        """Position of the node as an (x, y) pair."""
        return (self.x, self.y)

    @property
    def nodesIn(self) -> list[Node]:
        """Nodes connected to this node by an incoming edge."""
        return [edge.start for edge in self.edges_in]

    @property
    def nodesOut(self) -> list[Node]:
        """Nodes connected to this node by an outgoing edge."""
        return [edge.end for edge in self.edges_out]

    @property
    def neighbours(self) -> set[Node]:
        """Nodes connected to this node, excluding the node itself."""
        res = {e.end if e.start is self else e.start for e in self._edges}
        res.discard(self)
        return res

    # endregion

    # region cast
    def toDict(self) -> dict[str, object]:
        """Return a dictionary representation of the node.

        Returns:
            A dictionary containing the node data.
        """
        return {
            'key': self.key,
            'value': self.value,
            'position': {'x': self.x, 'y': self.y},
            'radius': self.radius,
            'diameter': self.diameter,
            'color': self.color,
            'edge_color': self.edge_color,
            'edge_width': self.edge_width,
            'properties': self.properties,
        }

    # endregion

    # region is
    def isSink(self) -> bool:
        """Return True if the node has no outgoing edge."""
        return self.degree_out == 0

    def isSource(self) -> bool:
        """Return True if the node has no incoming edge."""
        return self.degree_in == 0

    # endregion

    # region overwriting
    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and self.key == other.key

    def __str__(self) -> str:
        """Return a string representation of the node.

        Returns:
            A string summarizing the node.
        """
        return (
            f'{self.key} ({self.value}): Position ({self.x}, {self.y}), '
            f'Diameter {self.diameter}, '
            f'Color {self.color}, Border {self.edge_color} '
            f'(width {self.edge_width})'
        )

    # endregion
