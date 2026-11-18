"""A binary tree built on top of the tree, with left and right children."""

from __future__ import annotations

from .Node import Node
from .Tree import Tree


class BinaryTree(Tree):
    """A binary tree.

    A binary tree is a tree where every node has at most two children, a left
    one and a right one. The side of a child is stored in the properties of
    the edge that connects it to its parent.

    Args:
        name: Name of the tree, used in its string representation.
    """

    def __init__(self, name: str = 'undefined') -> None:
        super().__init__(name=name)

    def addChild(self, parent: Node, child: Node, **kwargs: object) -> None:
        """Raise an error, as children must be added with `addLeft` or `addRight`.

        Args:
            parent: Parent node.
            child: Child node.
            **kwargs: Arguments ignored, kept for signature compatibility.

        Raises:
            ValueError: Always, to direct the caller to `addLeft` or
                `addRight`.
        """
        raise ValueError('For BinaryTree, please use addLeft or addRight.')

    def addLeft(
        self,
        parent: Node,
        child: Node,
        weight: float = 1,
        color: str = 'black',
        width: float = 0.003,
        properties: dict[str, object] | None = None,
    ) -> None:
        """Add a left child to a parent node.

        Args:
            parent: Parent node already present in the tree.
            child: Child node to be added as the left child.
            weight: Weight of the edge.
            color: Color of the edge as drawn.
            width: Width of the edge as drawn.
            properties: Additional properties of the edge. A new empty
                dictionary is used when `None`.

        Raises:
            ValueError: If the parent already has a left child.
        """
        properties = dict(properties or {})
        properties['side'] = 'left'
        if self.getLeft(parent) is not None:
            raise ValueError('Left child already exists for this parent.')
        super().addChild(
            parent,
            child,
            weight=weight,
            color=color,
            width=width,
            properties=properties,
        )

    def addRight(
        self,
        parent: Node,
        child: Node,
        weight: float = 1,
        color: str = 'black',
        width: float = 0.003,
        properties: dict[str, object] | None = None,
    ) -> None:
        """Add a right child to a parent node.

        Args:
            parent: Parent node already present in the tree.
            child: Child node to be added as the right child.
            weight: Weight of the edge.
            color: Color of the edge as drawn.
            width: Width of the edge as drawn.
            properties: Additional properties of the edge. A new empty
                dictionary is used when `None`.

        Raises:
            ValueError: If the parent already has a right child.
        """
        properties = dict(properties or {})
        properties['side'] = 'right'
        if self.getRight(parent) is not None:
            raise ValueError('Right child already exists for this parent.')
        super().addChild(
            parent,
            child,
            weight=weight,
            color=color,
            width=width,
            properties=properties,
        )

    def getLeft(self, parent: Node) -> Node | None:
        """Return the left child of a parent node, or None if it has none.

        Args:
            parent: Parent node whose left child is looked up.

        Returns:
            The left child of the parent, or None if it has none.
        """
        for edge in parent.edges_out:
            if edge.properties.get('side') == 'left':
                return edge.end
        return None

    def getRight(self, parent: Node) -> Node | None:
        """Return the right child of a parent node, or None if it has none.

        Args:
            parent: Parent node whose right child is looked up.

        Returns:
            The right child of the parent, or None if it has none.
        """
        for edge in parent.edges_out:
            if edge.properties.get('side') == 'right':
                return edge.end
        return None

    def getChildren(self, parent: Node) -> list[Node]:
        """Return the children of a parent node, left first then right.

        Args:
            parent: Parent node whose children are looked up.

        Returns:
            The children of the parent, the left one first.
        """
        children: list[Node] = []
        left = self.getLeft(parent)
        right = self.getRight(parent)
        if left is not None:
            children.append(left)
        if right is not None:
            children.append(right)
        return children

    def __str__(self) -> str:
        """Return a string representation of the binary tree.

        Returns:
            A summary of the binary tree.
        """
        root_key = self.root.key if self.root else 'None'
        return f'BinaryTree (root: {root_key})\n' + super().__str__()
