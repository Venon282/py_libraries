"""A directed tree built on top of the general graph."""

from __future__ import annotations

from collections import deque
from collections.abc import Generator

from .Edge import Edge
from .Graph import Graph
from .Node import Node


class Tree(Graph):
    """A directed tree.

    A tree is a directed graph with a single root (in-degree 0). It reuses the
    node and edge management of the general graph and adds tree-specific
    operations.

    Args:
        name: Name of the tree, used in its string representation.
    """

    def __init__(self, name: str = 'undefined') -> None:
        super().__init__(name=name, is_directed=True)
        self._root: Node | None = None

    # region construction
    def addChild(
        self,
        parent: Node,
        child: Node,
        weight: float = 1,
        color: str = 'black',
        width: float = 0.003,
        properties: dict[str, object] | None = None,
    ) -> None:
        """Add a child node to a parent node in the tree.

        Creates the directed edge from the parent to the child.

        Args:
            parent: Parent node already present in the tree.
            child: Child node to be added.
            weight: Weight of the edge.
            color: Color of the edge as drawn.
            width: Width of the edge as drawn.
            properties: Additional properties of the edge. A new empty
                dictionary is used when `None`.

        Raises:
            ValueError: If the parent is missing or if the child is already
                in the tree.
        """
        if properties is None:
            properties = {}

        if parent.key not in self.nodes:
            raise ValueError(f'The parent node "{parent.key}" is not in the tree.')

        if child.key in self.nodes:
            raise ValueError(f'The child node "{child.key}" is already in the tree.')

        if self.hasPathDirected(child, parent):
            raise ValueError(
                f'Add "{child.key}" as child of "{parent.key}" would create a cycle.'
            )

        edge = Edge(
            start=parent,
            end=child,
            weight=weight,
            color=color,
            width=width,
            properties=properties,
        )

        self.addEdge(edge)

    def addChildByKey(
        self,
        parent_key: str,
        child_key: str,
        weight: float = 1,
        color: str = 'black',
        width: float = 0.003,
        properties: dict[str, object] | None = None,
    ) -> None:
        """Add a child to the tree by passing the keys directly.

        Creates the child node if it does not exist yet.

        Args:
            parent_key: Key of the parent node.
            child_key: Key of the child node.
            weight: Weight of the edge.
            color: Color of the edge as drawn.
            width: Width of the edge as drawn.
            properties: Additional properties of the edge. A new empty
                dictionary is used when `None`.

        Raises:
            ValueError: If no node is found for the parent key, or if the
                child key is already in the tree (see `addChild`).
        """
        parent = self.getNodeByKey(parent_key)

        if parent is None:
            raise ValueError(f'No node found with the key "{parent_key}"')

        child = self.getNodeByKey(child_key) or Node(child_key)
        self.addChild(
            parent,
            child,
            weight=weight,
            color=color,
            width=width,
            properties=properties,
        )

    # endregion

    # region is
    def isValidTree(self) -> bool:
        """Check that the current structure is a valid tree.

        The structure is valid when it has a defined root, n - 1 edges for n
        nodes, and every node is reachable from the root.

        Returns:
            True if the tree is valid, False otherwise.
        """
        if self.root is None:
            return False

        n = len(self.nodes)
        if n == 0:
            return False

        if len(self.edges) != n - 1:
            return False

        visited: set[str] = set()
        [_ for _ in self.dfsDirected(self.root, visited=visited)]
        return len(visited) == n

    def isParentOf(self, node_a: Node, node_b: Node) -> bool:
        """Return True if `node_a` is the parent of `node_b`."""
        return node_a is self.getParent(node_b)

    def isChildOf(self, node_a: Node, node_b: Node) -> bool:
        """Return True if `node_a` is a child of `node_b`."""
        return self.isParentOf(node_a=node_b, node_b=node_a)

    def isSiblingOf(self, node_a: Node, node_b: Node) -> bool:
        """Return True if `node_a` and `node_b` are siblings."""
        return any(sibling is node_b for sibling in self.getISiblings(node_a))

    def isAnscestorOf(self, node_a: Node, node_b: Node) -> bool:
        """Return True if `node_a` is an ancestor of `node_b`."""
        return any(ancestor is node_a for ancestor in self.getIAncestors(node_b))

    def isDescendantOf(self, node_a: Node, node_b: Node) -> bool:
        """Return True if `node_a` is a descendant of `node_b`."""
        return any(descendant is node_a for descendant in self.getIDescendants(node_b))

    def isLeaf(self, node: Node) -> bool:
        """Return True if the node has no child."""
        return len(node.edges_out) == 0

    # endregion

    # region get
    def getParent(self, node: Node) -> Node | None:
        """Return the parent of a node, or None if it is the root.

        Args:
            node: Node whose parent is looked up.

        Returns:
            The parent of the node, or None if it has no parent.
        """
        for edge in node.edges_in:
            return edge.start

        return None

    def getChildrens(self, node: Node) -> list[Node]:
        """Return the children of a node.

        Args:
            node: Node whose children are looked up.

        Returns:
            The children of the node.
        """
        return [edge.end for edge in node.edges_out]

    def getIChildrens(self, node: Node) -> Generator[Node, None, None]:
        """Iterate over the children of a node.

        Args:
            node: Node whose children are iterated.

        Yields:
            The children of the node.
        """
        for edge in node.edges_out:
            yield edge.end

    def getSiblings(self, node: Node) -> list[Node]:
        """Return the siblings of a node.

        Args:
            node: Node whose siblings are looked up.

        Returns:
            The siblings of the node, excluding the node itself.
        """
        parent = self.getParent(node)
        if parent is None:
            return []
        return [child for child in self.getChildrens(parent) if child is not node]

    def getISiblings(self, node: Node) -> Generator[Node, None, None]:
        """Iterate over the siblings of a node.

        Args:
            node: Node whose siblings are iterated.

        Yields:
            The siblings of the node, excluding the node itself.
        """
        parent = self.getParent(node)
        if parent is None:
            return

        for child in self.getIChildrens(parent):
            if child is node:
                continue

            yield child

    def getDepth(self, node: Node) -> int:
        """Return the depth of a node (distance from the root).

        The root is at depth 0.

        Args:
            node: Node whose depth is computed.

        Returns:
            The depth of the node.

        Raises:
            ValueError: If the root is not defined or the node is not
                connected to the root.
        """
        if self.root is None:
            raise ValueError('The root is not defined.')

        depth = 0
        current = node
        while current is not self.root:
            parent = self.getParent(current)
            if parent is None:
                raise ValueError('The node is not connected to the root.')

            current = parent
            depth += 1

        return depth

    def getHeight(self, node: Node | None = None) -> int:
        """Return the height of the tree or of a subtree.

        A leaf node has a height of 0.

        Args:
            node: Root of the subtree. The root of the tree is used when
                `None`.

        Returns:
            The height of the tree or of the subtree, or -1 if there is no
            root.
        """
        node = node or self.root
        if node is None:
            return -1

        que: deque[tuple[Node, int]] = deque([(node, 0)])
        max_height = 0

        while que:
            current, height = que.popleft()

            max_height = max(max_height, height)

            for children in self.getIChildrens(current):
                que.append((children, height + 1))

        return max_height

    def getLevel(self, depth: int) -> list[Node]:
        """Return all nodes at a given depth.

        Args:
            depth: Target level (0 is the root).

        Returns:
            The nodes at the given depth.
        """
        if self.root is None:
            return []

        result: list[Node] = []
        queue: deque[tuple[Node, int]] = deque([(self.root, 0)])

        while queue:
            node, d = queue.popleft()
            if d == depth:
                result.append(node)
            elif d < depth:
                for children in self.getIChildrens(node):
                    queue.append((children, d + 1))

        return result

    def getLeaves(self) -> list[Node]:
        """Return all the leaves of the tree.

        Returns:
            The nodes without children.
        """
        return [node for node in self.nodes.values() if node.degree_out == 0]

    def getAncestors(self, node: Node) -> list[Node]:
        """Return the ancestors of a node, ordered from the parent to the root.

        Args:
            node: Node whose ancestors are looked up.

        Returns:
            The ancestors of the node, ordered from the parent to the root.
        """
        ancestors: list[Node] = []
        current = self.getParent(node)
        while current is not None:
            ancestors.append(current)
            current = self.getParent(current)
        return ancestors

    def getIAncestors(self, node: Node) -> Generator[Node, None, None]:
        """Iterate over the ancestors of a node, from the parent to the root.

        Args:
            node: Node whose ancestors are iterated.

        Yields:
            The ancestors of the node, ordered from the parent to the root.
        """
        current = self.getParent(node)
        while current is not None:
            yield current
            current = self.getParent(current)

    def getDescendants(self, node: Node) -> list[Node]:
        """Return the descendants of a node, in breadth-first order.

        Args:
            node: Node whose descendants are looked up.

        Returns:
            The descendants of the node.
        """
        descendants: list[Node] = []
        queue = deque(self.getChildrens(node))
        visited: set[Node] = set()

        while queue:
            current = queue.popleft()
            if current in visited:
                continue

            visited.add(current)
            descendants.append(current)
            queue.extend(self.getChildrens(current))

        return descendants

    def getIDescendants(self, node: Node) -> Generator[Node, None, None]:
        """Iterate over the descendants of a node, in breadth-first order.

        Args:
            node: Node whose descendants are iterated.

        Yields:
            The descendants of the node.
        """
        queue = deque(self.getChildrens(node))
        visited: set[Node] = set()

        while queue:
            current = queue.popleft()
            if current in visited:
                continue

            visited.add(current)
            yield current
            queue.extend(self.getChildrens(current))

    def getLowestCommonAncestor(self, node_a: Node, node_b: Node) -> Node | None:
        """Return the lowest common ancestor of two nodes.

        Args:
            node_a: First node.
            node_b: Second node.

        Returns:
            The closest common ancestor of the two nodes, or None if there is
            none.
        """
        node_a_ancestors = set(self.getAncestors(node_a))

        for ancestor in self.getIAncestors(node_b):
            if ancestor in node_a_ancestors:
                return ancestor
        return None

    # endregion

    # region properties
    @property
    def root(self) -> Node | None:
        """Root of the tree, or None if it is not defined."""
        return self._root

    @root.setter
    def root(self, node: Node) -> None:
        """Set the root of the tree.

        If the node is not yet in the graph, it is added.

        Args:
            node: The root node.

        Raises:
            ValueError: If the node has an in-degree greater than 0.
        """
        if node.degree_in > 0:
            raise ValueError(
                f'The node "{node.key}" have parents so it can\'t be a root.'
            )

        self.addNode(node)
        self._root = node

    # endregion

    # region cast
    def toDict(self) -> dict[str, object]:
        """Return a dictionary representation of the tree.

        Extends the dictionary representation of the graph with the root key.

        Returns:
            A dictionary with the graph data and the root key.
        """
        base = super().toDict()
        base['root'] = self.root.key if self.root else None
        return base

    # endregion

    # region overwriting
    def __str__(self) -> str:
        """Return a string representation of the tree.

        Returns:
            A summary of the tree.
        """
        root_key = self.root.key if self.root else 'None'
        n = len(self.nodes)
        h = self.getHeight() if self.root else '?'
        leaves = len(self.getLeaves())
        return (
            f"--- Tree '{self.name}' ---\n"
            f'Root    : {root_key}\n'
            f'Nodes   : {n}  |  Edges : {len(self.edges)}\n'
            f'Height  : {h}  |  Leaves: {leaves}\n'
            f'Valid   : {self.isValidTree()}'
        )

    # endregion
