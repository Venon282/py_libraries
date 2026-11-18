"""An edge of a graph: a weighted, styled link between two nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .Node import Node

DEFAULT_WIDTH: float = 0.003
DEFAULT_COLOR: str = 'black'


class Edge:
    """A directed edge from a start node to an end node.

    Args:
        start: Node the edge starts from.
        end: Node the edge ends at.
        weight: Weight of the edge.
        properties: Additional properties of the edge. A new empty dictionary
            is used when `None`.
        color: Color of the edge as drawn.
        width: Width of the edge as drawn.
    """

    def __init__(
        self,
        start: Node,
        end: Node,
        weight: float = 1,
        properties: dict[str, object] | None = None,
        color: str = DEFAULT_COLOR,
        width: float = DEFAULT_WIDTH,
    ) -> None:
        self.start: Node = start
        self.end: Node = end
        self.weight: float = weight
        self.properties: dict[str, object] = (
            properties if properties is not None else {}
        )
        self.color: str = color
        self.width: float = width

    # region properties
    @property
    def dx(self) -> float:
        """Horizontal distance from the start node to the end node."""
        return self.end.x - self.start.x

    @property
    def dy(self) -> float:
        """Vertical distance from the start node to the end node."""
        return self.end.y - self.start.y

    @property
    def length(self) -> float:
        """Euclidean distance between the start and end nodes."""
        return float(np.hypot(self.dx, self.dy))

    @property
    def coordinate(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Endpoints of the edge trimmed to the node borders.

        Returns:
            A pair of (x, y) tuples: the trimmed start point and the trimmed
            end point.
        """
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

    # region cast
    def toDict(self) -> dict[str, object]:
        """Return a dictionary representation of the edge.

        Returns:
            A dictionary containing the edge data.
        """
        return {
            'start': self.start.key,
            'end': self.end.key,
            'weight': self.weight,
            'properties': self.properties,
        }

    # endregion

    # region overwriting
    def __hash__(self) -> int:
        return hash((self.start, self.end))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return False
        return self.start == other.start and self.end == other.end

    def __str__(self) -> str:
        """Return a string representation of the edge.

        Returns:
            A string summarizing the edge.
        """
        return (
            f'{self.start.key} -- {self.end.key}, '
            f'Weight {self.weight}, Color {self.color}, Width {self.width}'
        )

    # endregion
