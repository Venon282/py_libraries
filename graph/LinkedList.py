"""A singly linked list built on top of the tree."""

from __future__ import annotations

from collections.abc import Generator
from typing import TypedDict, Unpack

from .Edge import Edge
from .Node import Node
from .Tree import Tree


class EdgeKwargs(TypedDict, total=False):
    """Keyword arguments forwarded to the edge built by the list operations."""

    weight: float
    color: str
    width: float
    properties: dict[str, object] | None


class LinkedList(Tree):
    """A singly linked list.

    A linked list is a tree where every node has at most one child. It tracks
    the head (root) and the tail of the list.

    Args:
        name: Name of the list, used in its string representation.
    """

    def __init__(self, name: str = 'undefined') -> None:
        super().__init__(name=name)

        self._tail: Node | None = None

    # region properties
    @property
    def head(self) -> Node | None:
        """Head of the list, or None if it is empty."""
        return self.root

    @head.setter
    def head(self, node: Node) -> None:
        """Set the head of the list.

        Args:
            node: The head node.
        """
        self.root = node

    @property
    def tail(self) -> Node | None:
        """Tail of the list, or None if it is empty."""
        return self._tail

    @tail.setter
    def tail(self, node: Node) -> None:
        """Set the tail of the list.

        Args:
            node: The tail node.
        """
        self._tail = node

    # endregion

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
        """Add a child to a parent, enforcing the single-child constraint.

        Updates the tail when the child is appended at the end.

        Args:
            parent: Parent node already present in the list.
            child: Child node to be added.
            weight: Weight of the edge.
            color: Color of the edge as drawn.
            width: Width of the edge as drawn.
            properties: Additional properties of the edge. A new empty
                dictionary is used when `None`.

        Raises:
            ValueError: If the parent already has a child.
        """
        if parent.degree_out >= 1:
            raise ValueError(
                f'Node "{parent.key}" already has a child. '
                f'A LinkedList node can have at most one successor.'
            )
        super().addChild(
            parent,
            child,
            weight=weight,
            color=color,
            width=width,
            properties=properties,
        )

        # child has no successor -> it becomes the new tail
        if child.degree_out == 0:
            self._tail = child

    def append(self, node: Node, **kwargs: Unpack[EdgeKwargs]) -> None:
        """Append a node at the end of the list (after the tail).

        Args:
            node: Node to append. It must not already be in the list.
            **kwargs: Arguments forwarded to the edge built between the tail
                and the node.

        Raises:
            ValueError: If the node is already in the list.
        """
        if node.key in self.nodes:
            raise ValueError(f'Node "{node.key}" is already in the list.')

        if self._tail is None:
            # Empty list so node becomes both head and tail
            self.addNode(node)
            self._root = node
            self._tail = node
        else:
            # _tail updated inside addChild
            self.addChild(self._tail, node, **kwargs)

    def appendByKey(self, key: str, **kwargs: Unpack[EdgeKwargs]) -> None:
        """Append a new node built from a key at the end of the list.

        Args:
            key: Key of the node to append.
            **kwargs: Arguments forwarded to the edge built between the tail
                and the node.
        """
        self.append(Node(key), **kwargs)

    def prepend(self, node: Node, **kwargs: Unpack[EdgeKwargs]) -> None:
        """Insert a node at the beginning of the list (before the head).

        The new node becomes the new head.

        Args:
            node: Node to prepend. It must not already be in the list.
            **kwargs: Arguments forwarded to the edge built between the node
                and the old head.

        Raises:
            ValueError: If the node is already in the list.
        """
        if node.key in self.nodes:
            raise ValueError(f'Node "{node.key}" is already in the list.')

        old_head = self.head

        if old_head is None:
            # Empty list
            self.addNode(node)
            self._root = node
            self._tail = node
        else:
            self.addNode(node)
            self.addEdge(Edge(start=node, end=old_head, **kwargs))
            self._root = node

    def prependByKey(self, key: str, **kwargs: Unpack[EdgeKwargs]) -> None:
        """Prepend a new node built from a key at the beginning of the list.

        Args:
            key: Key of the node to prepend.
            **kwargs: Arguments forwarded to the edge built between the node
                and the old head.
        """
        self.prepend(Node(key), **kwargs)

    def insertAfter(self, ref: Node, node: Node, **kwargs: Unpack[EdgeKwargs]) -> None:
        """Insert a node immediately after a reference node.

        Before: ref -> next_node
        After:  ref -> node -> next_node

        Args:
            ref: Reference node already in the list.
            node: Node to insert. It must not already be in the list.
            **kwargs: Arguments forwarded to the edges rebuilt around the
                node.

        Raises:
            ValueError: If the reference node is not in the list or the node
                is already in it.
        """
        if ref.key not in self.nodes:
            raise ValueError(f'Reference node "{ref.key}" is not in the list.')
        if node.key in self.nodes:
            raise ValueError(f'Node "{node.key}" is already in the list.')

        next_node = self.getNext(ref)

        # Disconnect ref -> next_node
        if next_node is not None:
            old_edge = next((e for e in ref.edges_out if e.end is next_node), None)
            if old_edge is not None:
                self.removeEdge(old_edge)

        # Connect ref -> node -> next_node
        self.addNode(node)
        self.addEdge(Edge(start=ref, end=node, **kwargs))

        if next_node is not None:
            self.addEdge(Edge(start=node, end=next_node, **kwargs))
        else:
            # ref was the tail -> node is the new tail
            self._tail = node

    def insertAfterByKey(
        self, ref_key: str, new_key: str, **kwargs: Unpack[EdgeKwargs]
    ) -> None:
        """Insert a new node built from a key after a reference node.

        Args:
            ref_key: Key of the reference node.
            new_key: Key of the node to insert.
            **kwargs: Arguments forwarded to the edges rebuilt around the
                node.

        Raises:
            ValueError: If no node is found for the reference key.
        """
        ref = self.getNodeByKey(ref_key)
        if ref is None:
            raise ValueError(f'No node found with key "{ref_key}".')
        self.insertAfter(ref, Node(new_key), **kwargs)

    def insertBefore(self, ref: Node, node: Node, **kwargs: Unpack[EdgeKwargs]) -> None:
        """Insert a node immediately before a reference node.

        Before: prev_node -> ref
        After:  prev_node -> node -> ref

        Args:
            ref: Reference node already in the list.
            node: Node to insert. It must not already be in the list.
            **kwargs: Arguments forwarded to the edges rebuilt around the
                node.

        Raises:
            ValueError: If the reference node is not in the list or the node
                is already in it.
        """
        if ref.key not in self.nodes:
            raise ValueError(f'Reference node "{ref.key}" is not in the list.')
        if node.key in self.nodes:
            raise ValueError(f'Node "{node.key}" is already in the list.')

        prev_node = self.getPrev(ref)

        if prev_node is None:
            # ref is the head, prepend
            self.prepend(node, **kwargs)
        else:
            self.insertAfter(prev_node, node, **kwargs)

    def insertBeforeByKey(
        self, ref_key: str, new_key: str, **kwargs: Unpack[EdgeKwargs]
    ) -> None:
        """Insert a new node built from a key before a reference node.

        Args:
            ref_key: Key of the reference node.
            new_key: Key of the node to insert.
            **kwargs: Arguments forwarded to the edges rebuilt around the
                node.

        Raises:
            ValueError: If no node is found for the reference key.
        """
        ref = self.getNodeByKey(ref_key)
        if ref is None:
            raise ValueError(f'No node found with key "{ref_key}".')
        self.insertBefore(ref, Node(new_key), **kwargs)

    def remove(self, node: Node) -> None:
        """Remove a node from the list.

        Reconnects the predecessor of the node to its successor.

        Before: prev -> node -> next
        After:  prev -> next

        Args:
            node: Node to remove.

        Raises:
            ValueError: If the node is not in the list.
        """
        if node.key not in self.nodes:
            raise ValueError(f'Node "{node.key}" is not in the list.')

        prev_node = self.getPrev(node)
        next_node = self.getNext(node)

        self.removeNode(node)

        # Reconnect if both sides exist
        if prev_node is not None and next_node is not None:
            self.addEdge(Edge(start=prev_node, end=next_node))

        # Update head / tail if needed
        if node is self._root:
            self._root = next_node

        if node is self._tail:
            self._tail = prev_node

    def removeByKey(self, key: str) -> None:
        """Remove the node with the given key from the list.

        Args:
            key: Key of the node to remove.

        Raises:
            ValueError: If no node is found with the given key.
        """
        node = self.getNodeByKey(key)
        if node is None:
            raise ValueError(f'No node found with key "{key}".')
        self.remove(node)

    # endregion

    # region is
    def isHead(self, node: Node) -> bool:
        """Return True if the node is the head of the list."""
        return node is self._root

    def isTail(self, node: Node) -> bool:
        """Return True if the node is the tail of the list."""
        return node is self._tail

    def isEmpty(self) -> bool:
        """Return True if the list contains no node."""
        return len(self.nodes) == 0

    # endregion

    # region get
    def getNext(self, node: Node) -> Node | None:
        """Return the successor of a node, or None if it is the tail.

        Args:
            node: Node whose successor is looked up.

        Returns:
            The successor of the node, or None if it is the tail.
        """
        for edge in node.edges_out:
            return edge.end
        return None

    def getPrev(self, node: Node) -> Node | None:
        """Return the predecessor of a node, or None if it is the head.

        Args:
            node: Node whose predecessor is looked up.

        Returns:
            The predecessor of the node, or None if it is the head.
        """
        return self.getParent(node)

    def getNodeAt(self, index: int) -> Node | None:
        """Return the node at a given position, counting from the head.

        Args:
            index: Zero-based position of the node.

        Returns:
            The node at the given position, or None if it is out of range.

        Raises:
            ValueError: If the index is negative.
        """
        if index < 0:
            raise ValueError('Index must be >= 0.')

        current = self.head
        for _ in range(index):
            if current is None:
                return None
            current = self.getNext(current)

        return current

    def getINodes(self, start: Node | None = None) -> Generator[Node, None, None]:
        """Iterate over all the nodes from a start node to the tail.

        Args:
            start: Node to start from. The head is used when `None`.

        Yields:
            The nodes from the start node to the tail.
        """
        current = start or self.head
        while current is not None:
            yield current
            current = self.getNext(current)

    def getNodes(self, start: Node | None = None) -> list[Node]:
        """Return the ordered list of nodes from a start node to the tail.

        Args:
            start: Node to start from. The head is used when `None`.

        Returns:
            The nodes from the start node to the tail.
        """
        return list(self.getINodes(start))

    # endregion

    # region cast
    def toList(self) -> list[str]:
        """Return the keys of the nodes in order from the head to the tail.

        Returns:
            The keys of the nodes, in list order.
        """
        return [node.key for node in self.getINodes()]

    def toDict(self) -> dict[str, object]:
        """Return a dictionary representation of the list.

        Extends the dictionary representation of the tree with the head and
        tail keys.

        Returns:
            A dictionary with the tree data and the head and tail keys.
        """
        base = super().toDict()
        base['head'] = self.head.key if self.head else None
        base['tail'] = self.tail.key if self.tail else None
        return base

    # endregion

    # region overwriting
    def __str__(self) -> str:
        """Return a string representation of the list.

        Returns:
            A summary of the list.
        """
        head_key = self.head.key if self.head else 'None'
        tail_key = self.tail.key if self.tail else 'None'
        chain = ' -> '.join(self.toList()) if self.head else '(empty)'
        return (
            f"--- LinkedList '{self.name}' ---\n"
            f'Head  : {head_key}  |  Tail : {tail_key}\n'
            f'Length: {len(self)}\n'
            f'Chain : {chain}'
        )

    # endregion
