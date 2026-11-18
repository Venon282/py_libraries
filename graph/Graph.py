"""A general directed or undirected graph of nodes and edges, with drawing."""

from __future__ import annotations

from collections import deque
from collections.abc import Generator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from .Edge import Edge
from .Node import Node

DEFAULT_NAME: str = 'undefined'
DEFAULT_SHOW_LABELS_MAX_NODES: int = 150
DEFAULT_FIG_SIZE: tuple[float, float] = (10, 8)


class Graph:
    """A directed or undirected graph of nodes and edges.

    Args:
        name: Name of the graph, used in its string representation.
        is_directed: Whether the graph is directed.
        show_labels_max_nodes: Maximum number of visible nodes for which the
            node labels are shown when drawing.
    """

    def __init__(
        self,
        name: str = DEFAULT_NAME,
        is_directed: bool = True,
        show_labels_max_nodes: int = DEFAULT_SHOW_LABELS_MAX_NODES,
    ) -> None:
        self.name: str = name
        self.nodes: dict[str, Node] = {}
        self.edges: set[Edge] = set()
        self.is_directed: bool = is_directed
        self.show_labels_max_nodes: int = show_labels_max_nodes

    # region construct
    @staticmethod
    def constructByKeyPairs(
        pairs: list[tuple[object, object]],
    ) -> Graph:
        """Build a directed graph from pairs of node keys.

        Args:
            pairs: Pairs of keys. Each pair creates an edge between the two
                nodes, which are created on the fly.

        Returns:
            A new directed graph built from the pairs.
        """

        def _getNode(key: str, graph: Graph) -> Node:
            if key in graph.nodes:
                return graph.nodes[key]
            node = Node(key)
            graph.nodes[key] = node
            return node

        graph = Graph()
        for key1, key2 in pairs:
            node1 = _getNode(str(key1), graph)
            node2 = _getNode(str(key2), graph)
            edge = Edge(node1, node2)
            graph.edges.add(edge)
            node1.addEdge(edge)
            node2.addEdge(edge)

        return graph

    # endregion

    # region updates
    def addNode(self, node: Node) -> None:
        """Add a node to the graph if its key is not already present.

        Args:
            node: Node to add.
        """
        if node.key not in self.nodes:
            self.nodes[node.key] = node

    def removeNode(self, node: Node) -> None:
        """Remove a node from the graph, along with its connected edges.

        Args:
            node: Node to remove. If it is not in the graph, nothing happens.
        """
        if self.nodes.pop(node.key, None) is None:
            return

        self.edges = self.edges - node.edges

        for edge in node.edges:
            neighbor = edge.end if edge.start is node else edge.start
            if neighbor is not node:
                neighbor.discardEdge(edge)

        node.edges.clear()

    def addEdge(self, edge: Edge) -> None:
        """Add an edge to the graph and register it with both nodes.

        Args:
            edge: Edge to add. Its start and end nodes are added to the graph
                if they are not already present.
        """
        self.addNode(edge.start)
        self.addNode(edge.end)

        self.edges.add(edge)

        edge.start.addEdge(edge)
        edge.end.addEdge(edge)

    def removeEdge(self, edge: Edge) -> None:
        """Remove an edge from the graph and from both nodes.

        Args:
            edge: Edge to remove.

        Raises:
            KeyError: If the edge is not in the graph.
        """
        self.edges.remove(edge)
        edge.start.removeEdge(edge)
        edge.end.removeEdge(edge)

    def discardEdge(self, edge: Edge) -> None:
        """Discard an edge from the graph and from both nodes if present.

        Args:
            edge: Edge to discard.
        """
        self.edges.discard(edge)
        edge.start.discardEdge(edge)
        edge.end.discardEdge(edge)

    # endregion

    # region utils
    def bfsDirected(
        self, start: Node | None, visited: set[str] | None = None
    ) -> Generator[Node, None, None]:
        """Breadth-first search from a start node on a directed graph.

        Args:
            start: Node to start from. Nothing is yielded when `None`.
            visited: Keys of nodes already visited. A new set is used when
                `None`.

        Yields:
            The nodes in breadth-first order.
        """
        if start is None:
            return

        if visited is None:
            visited = set()
        queue = deque([start])

        while queue:
            node = queue.popleft()

            if node.key in visited:
                continue

            visited.add(node.key)

            yield node

            for edge in node.edges_out:
                if edge.end.key not in visited:
                    queue.append(edge.end)

    def bfsNonDirected(
        self, start: Node | None, visited: set[str] | None = None
    ) -> Generator[Node, None, None]:
        """Breadth-first search from a start node on an undirected graph.

        Args:
            start: Node to start from. Nothing is yielded when `None`.
            visited: Keys of nodes already visited. A new set is used when
                `None`.

        Yields:
            The nodes in breadth-first order.
        """
        if start is None:
            return

        if visited is None:
            visited = set()
        queue = deque([start])

        while queue:
            node = queue.popleft()

            if node.key in visited:
                continue

            visited.add(node.key)

            yield node

            for edge in node.edges_out:
                if edge.end.key not in visited:
                    queue.append(edge.end)

            for edge in node.edges_in:
                if edge.start.key not in visited:
                    queue.append(edge.start)

    def bfs(
        self, start: Node | None, visited: set[str] | None = None
    ) -> Generator[Node, None, None]:
        """Breadth-first search from a start node.

        The search is directed or undirected according to the graph.

        Args:
            start: Node to start from. Nothing is yielded when `None`.
            visited: Keys of nodes already visited. A new set is used when
                `None`.

        Yields:
            The nodes in breadth-first order.
        """
        if self.is_directed:
            yield from self.bfsDirected(start=start, visited=visited)
        else:
            yield from self.bfsNonDirected(start=start, visited=visited)

    def _dfsGenDirected(
        self, node: Node, visited: set[str], order: str
    ) -> Generator[Node, None, None]:
        if node.key in visited:
            return

        visited.add(node.key)

        if order == 'pre':
            yield node

        for edge in node.edges_out:
            yield from self._dfsGenDirected(edge.end, visited, order)

        if order == 'post':
            yield node

    def dfsDirected(
        self,
        start: Node | None = None,
        order: str = 'pre',
        visited: set[str] | None = None,
    ) -> Generator[Node, None, None]:
        """Depth-first search from a start node on a directed graph.

        Args:
            start: Node to start from. Nothing is yielded when `None`.
            order: 'pre' for pre-order (node before its successors) or 'post'
                for post-order (node after its successors).
            visited: Keys of nodes already visited. A new set is used when
                `None`.

        Yields:
            The nodes in the chosen depth-first order.
        """
        if start is None:
            return

        if visited is None:
            visited = set()

        yield from self._dfsGenDirected(start, visited, order)

    def _dfsGenNonDirected(
        self, node: Node, visited: set[str], order: str
    ) -> Generator[Node, None, None]:
        if node.key in visited:
            return

        visited.add(node.key)

        if order == 'pre':
            yield node

        for edge in node.edges_out:
            yield from self._dfsGenNonDirected(edge.end, visited, order)

        for edge in node.edges_in:
            yield from self._dfsGenNonDirected(edge.start, visited, order)

        if order == 'post':
            yield node

    def dfsNonDirected(
        self,
        start: Node | None = None,
        order: str = 'pre',
        visited: set[str] | None = None,
    ) -> Generator[Node, None, None]:
        """Depth-first search from a start node on an undirected graph.

        Args:
            start: Node to start from. Nothing is yielded when `None`.
            order: 'pre' for pre-order (node before its successors) or 'post'
                for post-order (node after its successors).
            visited: Keys of nodes already visited. A new set is used when
                `None`.

        Yields:
            The nodes in the chosen depth-first order.
        """
        if start is None:
            return

        if visited is None:
            visited = set()

        yield from self._dfsGenNonDirected(start, visited, order)

    def dfs(
        self,
        start: Node | None = None,
        order: str = 'pre',
        visited: set[str] | None = None,
    ) -> Generator[Node, None, None]:
        """Depth-first search from a start node.

        The search is directed or undirected according to the graph.

        Args:
            start: Node to start from. Nothing is yielded when `None`.
            order: 'pre' for pre-order (node before its successors) or 'post'
                for post-order (node after its successors).
            visited: Keys of nodes already visited. A new set is used when
                `None`.

        Yields:
            The nodes in the chosen depth-first order.
        """
        if self.is_directed:
            yield from self.dfsDirected(start=start, order=order, visited=visited)
        else:
            yield from self.dfsNonDirected(start=start, order=order, visited=visited)

    # endregion

    # region is
    def isLinkedList(self) -> bool:
        """Return True if the graph is a single directed chain.

        The graph must have exactly one head (in-degree 0, out-degree 1),
        exactly one tail (in-degree 1, out-degree 0), only middle nodes
        (in-degree 1, out-degree 1) in between, and all nodes must belong to
        the same chain.

        Returns:
            True if the graph is a linked list, False otherwise.
        """
        num_nodes = len(self.nodes)

        if num_nodes == 0:
            return False

        if num_nodes == 1:
            return len(next(iter(self.nodes.values())).edges) == 0

        head_node: Node | None = None
        tail_node: Node | None = None

        for node in self.nodes.values():
            out_degree = node.degree_out
            in_degree = node.degree_in

            if out_degree == 1 and in_degree == 0:
                if head_node is not None:
                    return False
                head_node = node
            elif out_degree == 0 and in_degree == 1:
                if tail_node is not None:
                    return False
                tail_node = node
            elif out_degree == 1 and in_degree == 1:
                continue
            else:
                return False

        if head_node is None or tail_node is None:
            return False

        visited_count = 0
        curr = head_node
        while curr is not None:
            visited_count += 1
            next_edge = next((e for e in curr.edges if e.start is curr), None)
            curr = next_edge.end if next_edge is not None else None

        return visited_count == num_nodes

    def isTree(self) -> bool:
        """Return True if the graph is a directed tree.

        The graph must have a unique root (in-degree 0), at most one parent
        per node, and every node must be reachable from the root without
        revisiting a node.

        Returns:
            True if the graph is a tree, False otherwise.
        """
        n_nodes = len(self.nodes)

        if n_nodes == 0:
            return False
        if n_nodes == 1:
            return next(iter(self.nodes.values())).degree_in == 0

        root: Node | None = None
        for node in self.nodes.values():
            if node.degree_in > 1:
                return False
            elif node.degree_in == 0:
                if root is not None:
                    return False
                root = node

        if root is None:
            return False

        queue = deque([root])
        visited = {root.key}

        while queue:
            node = queue.popleft()

            for edge in node.edges_out:
                neighbor = edge.end

                if neighbor.key not in visited:
                    visited.add(neighbor.key)
                    queue.append(neighbor)
                else:
                    return False

        return len(visited) == n_nodes

    def isRegularGrap(self) -> bool:
        """Return True if every node has the same in- and out-degree.

        Returns:
            True if the graph is regular, False otherwise. An empty graph is
            considered regular.
        """
        if not self.nodes:
            return True

        first_node = next(iter(self.nodes.values()))
        ref_in = first_node.degree_in
        ref_out = first_node.degree_out

        for node in self.nodes.values():
            if node.degree_in != ref_in or node.degree_out != ref_out:
                return False

        return True

    def isCompleteGraph(self) -> bool:
        """Return True if every node is connected to every other node.

        Each node must have an in- and out-degree of n - 1 and no self-loop.

        Returns:
            True if the graph is complete, False otherwise.
        """
        n_nodes = len(self.nodes)
        if n_nodes <= 1:
            return True

        expected_degree = n_nodes - 1

        for node in self.nodes.values():
            if node.degree_out != expected_degree or node.degree_in != expected_degree:
                return False

            for edge in node.edges_out:
                if edge.end is node:
                    return False

        return True

    # endregion

    # region has
    def hasPathNonDirected(self, source: Node, target: Node) -> bool:
        """Return True if a path exists between two nodes, ignoring direction.

        Args:
            source: Node to start from.
            target: Node to reach.

        Returns:
            True if a path exists from `source` to `target`.
        """
        visited = set()
        queue = deque([source])

        while queue:
            current = queue.popleft()

            if current is target:
                return True

            if current.key in visited:
                continue

            visited.add(current.key)

            for edge in current.edges_out:
                queue.append(edge.end)

            for edge in current.edges_in:
                queue.append(edge.start)

        return False

    def hasPathDirected(self, source: Node, target: Node) -> bool:
        """Return True if a directed path exists from a source to a target.

        Args:
            source: Node to start from.
            target: Node to reach.

        Returns:
            True if a directed path exists from `source` to `target`.
        """
        visited = set()
        queue = deque([source])

        while queue:
            current = queue.popleft()

            if current is target:
                return True

            if current.key in visited:
                continue

            visited.add(current.key)

            for edge in current.edges_out:
                queue.append(edge.end)

        return False

    def hasPath(self, source: Node, target: Node) -> bool:
        """Return True if a path exists between two nodes.

        The search is directed or undirected according to the graph.

        Args:
            source: Node to start from.
            target: Node to reach.

        Returns:
            True if a path exists from `source` to `target`.
        """
        if self.is_directed:
            return self.hasPathDirected(source=source, target=target)
        else:
            return self.hasPathNonDirected(source=source, target=target)

    # endregion

    # region get
    def getEdgesPosition(
        self,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Compute the trimmed start and end points of every edge.

        Each edge is shortened so that it starts and ends at the border of the
        two nodes it connects, instead of at their centers.

        Returns:
            A pair of (x, y) array tuples: the trimmed start points and the
            trimmed end points, one per edge.
        """
        start_xs = self.edgesStartX
        start_ys = self.edgesStartY
        end_xs = self.edgesEndX
        end_ys = self.edgesEndY
        start_radius = self.edgesStartRadius
        end_radius = self.edgesEndRadius

        dxs = end_xs - start_xs
        dys = end_ys - start_ys

        distance = np.hypot(dxs, dys)
        distance_mask = distance == 0
        distance[distance_mask] = 1

        x_normalisation = dxs / distance
        y_normalisation = dys / distance

        start_x = start_xs + x_normalisation * start_radius
        start_y = start_ys + y_normalisation * start_radius

        end_x = end_xs - x_normalisation * end_radius
        end_y = end_ys - y_normalisation * end_radius

        return (start_x, start_y), (end_x, end_y)

    def getNodeByKey(self, key: str) -> Node | None:
        """Return the node with the given key, or None if it is not present.

        Args:
            key: Key of the node to look up.

        Returns:
            The node with the matching key, or None if not found.
        """
        return self.nodes.get(key, None)

    # endregion

    # region visualisation
    def drawNodes(self, fig: Figure, ax: Axes) -> None:
        """Draw the nodes and their labels on the given axes.

        The labels are only shown when the number of visible nodes is below
        `show_labels_max_nodes`, and are updated whenever the axes limits
        change.

        Args:
            fig: Figure the axes belong to.
            ax: Axes to draw the nodes on.
        """

        def updateLabelVisibility(event: object = None) -> None:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()

            visible_nodes = {
                node.key
                for node, _txt in texts
                if x0 <= node.x <= x1 and y0 <= node.y <= y1
            }

            show = len(visible_nodes) <= self.show_labels_max_nodes

            for node, txt in texts:
                txt.set_visible(show and node.key in visible_nodes)

            fig.canvas.draw_idle()

        patches = [
            Circle((node.x, node.y), node.radius) for node in self.nodes.values()
        ]
        node_colors = self.nodesColor
        edge_colors = self.nodesEdgeColor
        edge_widths = self.nodesEdgeWidth
        texts = [
            (
                node,
                ax.text(
                    node.x,
                    node.y,
                    str(node.key),
                    fontsize=8,
                    ha='center',
                    va='center',
                    zorder=3,
                    fontweight='bold',
                    visible=False,
                ),
            )
            for node in self.nodes.values()
        ]

        collection = PatchCollection(
            patches,
            facecolor=node_colors,
            edgecolor=edge_colors,
            linewidths=edge_widths,
        )

        ax.add_collection(collection)

        ax.callbacks.connect('xlim_changed', updateLabelVisibility)
        ax.callbacks.connect('ylim_changed', updateLabelVisibility)
        updateLabelVisibility()

    def drawEdges(self, ax: Axes) -> None:
        """Draw the edges on the given axes.

        Directed graphs are drawn with arrows, undirected graphs with line
        segments.

        Args:
            ax: Axes to draw the edges on.
        """
        (
            (edge_start_x, edge_start_y),
            (
                edge_end_x,
                edge_end_y,
            ),
        ) = self.getEdgesPosition()

        if self.is_directed:
            dx = edge_end_x - edge_start_x
            dy = edge_end_y - edge_start_y

            ax.quiver(
                edge_start_x,
                edge_start_y,
                dx,
                dy,
                angles='xy',
                scale_units='xy',
                scale=1,
                width=next(iter(self.edges)).width,
                color=self.edgesColor,
            )
        else:
            starts = np.stack((edge_start_x, edge_start_y), axis=1)
            ends = np.stack((edge_end_x, edge_end_y), axis=1)

            segments = np.stack((starts, ends), axis=1)

            lc = LineCollection(
                segments, colors=self.edgesColor, linewidths=self.edgesWidth
            )
            ax.add_collection(lc)

    def draw(self, fig_size: tuple[float, float] = DEFAULT_FIG_SIZE) -> None:
        """Draw the graph in a new figure and display it.

        Args:
            fig_size: Size of the figure as a (width, height) pair.
        """

        def defineLimits(ax: Axes) -> None:
            xs = self.nodesX
            ys = self.nodesY
            rs = self.nodesRadius

            min_x = np.min(xs - rs)
            max_x = np.max(xs + rs)
            min_y = np.min(ys - rs)
            max_y = np.max(ys + rs)

            x_margin = (max_x - min_x) * 0.01
            y_margin = (max_y - min_y) * 0.01
            ax.set_xlim(min_x - x_margin, max_x + x_margin)
            ax.set_ylim(min_y - y_margin, max_y + y_margin)

        fig, ax = plt.subplots(figsize=fig_size)

        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.axis('off')

        self.drawNodes(fig, ax)
        self.drawEdges(ax)

        defineLimits(ax)

        plt.tight_layout()
        plt.show()

    # endregion

    # region properties
    @property
    def nodesX(self) -> np.ndarray:
        """X coordinates of the nodes."""
        return np.array([n.x for n in self.nodes.values()])

    @property
    def nodesY(self) -> np.ndarray:
        """Y coordinates of the nodes."""
        return np.array([n.y for n in self.nodes.values()])

    @property
    def nodesRadius(self) -> np.ndarray:
        """Radii of the nodes."""
        return np.array([n.radius for n in self.nodes.values()])

    @property
    def nodesColor(self) -> np.ndarray:
        """Fill colors of the nodes."""
        return np.array([n.color for n in self.nodes.values()])

    @property
    def nodesEdgeColor(self) -> np.ndarray:
        """Border colors of the nodes."""
        return np.array([n.edge_color for n in self.nodes.values()])

    @property
    def nodesEdgeWidth(self) -> np.ndarray:
        """Border widths of the nodes."""
        return np.array([n.edge_width for n in self.nodes.values()])

    @property
    def edgesStartX(self) -> np.ndarray:
        """X coordinates of the start nodes of the edges."""
        return np.array([e.start.x for e in self.edges])

    @property
    def edgesStartY(self) -> np.ndarray:
        """Y coordinates of the start nodes of the edges."""
        return np.array([e.start.y for e in self.edges])

    @property
    def edgesStartRadius(self) -> np.ndarray:
        """Radii of the start nodes of the edges."""
        return np.array([e.start.radius for e in self.edges])

    @property
    def edgesEndX(self) -> np.ndarray:
        """X coordinates of the end nodes of the edges."""
        return np.array([e.end.x for e in self.edges])

    @property
    def edgesEndY(self) -> np.ndarray:
        """Y coordinates of the end nodes of the edges."""
        return np.array([e.end.y for e in self.edges])

    @property
    def edgesEndRadius(self) -> np.ndarray:
        """Radii of the end nodes of the edges."""
        return np.array([e.end.radius for e in self.edges])

    @property
    def edgesDX(self) -> np.ndarray:
        """Horizontal distances of the edges."""
        return self.edgesEndX - self.edgesStartX

    @property
    def edgesDY(self) -> np.ndarray:
        """Vertical distances of the edges."""
        return self.edgesEndY - self.edgesStartY

    @property
    def edgesCoordinate(self) -> np.ndarray:
        """Trimmed start and end points of the edges."""
        return np.array([e.coordinate for e in self.edges])

    @property
    def edgesColor(self) -> np.ndarray:
        """Colors of the edges."""
        return np.array([e.color for e in self.edges])

    @property
    def edgesWidth(self) -> np.ndarray:
        """Widths of the edges."""
        return np.array([e.width for e in self.edges])

    # endregion

    # region cast
    def toDict(self) -> dict[str, object]:
        """Return a dictionary representation of the graph.

        Returns:
            A dictionary with lists of nodes and edges.
        """
        return {
            'nodes': [node.toDict() for node in self.nodes.values()],
            'edges': [edge.toDict() for edge in self.edges],
        }

    def toList(self) -> list[tuple[str, str]]:
        """Return the edges of the graph as pairs of node keys.

        Returns:
            A list of (start_key, end_key) tuples, one per edge.
        """
        return [(edge.start.key, edge.end.key) for edge in self.edges]

    # endregion

    # region overwriting
    def __contains__(self, node: Node) -> bool:
        """Return True if the graph contains the given node."""
        return node.key in self.nodes

    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self.nodes)

    def __iter__(self) -> Generator[Node, None, None]:
        """Iterate over the nodes of the graph.

        Yields:
            The nodes of the graph.
        """
        yield from self.nodes.values()

    def __str__(self) -> str:
        """Return a string representation of the graph.

        Returns:
            A summary of the graph's nodes and edges.
        """
        nodes_str = ', '.join(self.nodes.keys())
        edges_str = ', '.join(
            [
                f'{edge.start.key}->{edge.end.key}'
                if self.is_directed
                else f'{edge.start.key}-{edge.end.key}'
                for edge in self.edges
            ]
        )
        return f'--- Graph {self.name} ---\nNodes: {nodes_str}\nEdges: {edges_str}'

    # endregion
