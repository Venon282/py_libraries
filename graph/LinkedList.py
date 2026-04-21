from collections import deque

# internal
from .Tree import Tree
from .Edge import Edge
from .Node import Node

class LinkedList(Tree):
    def __init__(self, name='undefined'):
        """
        Initializes an empty LinkedList.
        A LinkedList is a Tree where every node has at most one child.
        """
        super().__init__(name=name)
        
        self._tail: Node | None = None
        
    # region properties
    @property
    def head(self) -> Node | None:
        return self.root

    @head.setter
    def head(self, node: Node):
        self.root = node

    @property
    def tail(self) -> Node | None:
        return self._tail

    @tail.setter
    def tail(self, node: Node):
        self._tail = node
    # end region
    
    # region construction
    def addChild(self, parent: Node, child: Node,
                 weight=1, color='black', width=0.003, properties={}):
        """
        Overrides Tree.addChild to enforce the single-child constraint.
        Updates _tail if the child is appended at the end.

        Raises:
            ValueError: If the parent already has a child.
        """
        if parent.degree_out >= 1:
            raise ValueError(
                f'Node "{parent.key}" already has a child. '
                f'A LinkedList node can have at most one successor.'
            )
        super().addChild(parent, child,
                         weight=weight, color=color,
                         width=width, properties=properties)

        # child has no successor -> it becomes the new tail
        if child.degree_out == 0:
            self._tail = child
            
    def append(self, node: Node, **kwargs):
        """
        Appends a node at the end of the list (after the tail).

        Parameters:
            node (Node): Node to append. Must not already be in the list.

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

    def appendByKey(self, key: str, **kwargs):
        """
        Appends a new node built from 'key' at the end of the list.
        """
        self.append(Node(key), **kwargs)
    
    def prepend(self, node: Node, **kwargs):
        """
        Inserts a node at the beginning of the list (before the head).
        The new node becomes the new head.

        Parameters:
            node (Node): Node to prepend. Must not already be in the list.

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

    def prependByKey(self, key: str, **kwargs):
        """
        Prepends a new node built from 'key' at the beginning of the list.
        """
        self.prepend(Node(key), **kwargs)

    def insertAfter(self, node: Node, ref: Node, **kwargs):
        """
        Inserts 'node' immediately after 'ref'.

        Before: ref -> next_node
        After:  ref -> node -> next_node

        Parameters:
            ref  (Node): Reference node already in the list.
            node (Node): Node to insert. Must not already be in the list.

        Raises:
            ValueError: If 'ref' is not in the list or 'node' is already in it.
        """
        if ref.key not in self.nodes:
            raise ValueError(f'Reference node "{ref.key}" is not in the list.')
        if node.key in self.nodes:
            raise ValueError(f'Node "{node.key}" is already in the list.')

        next_node = self.getNext(ref)

        # Disconnect ref -> next_node
        if next_node is not None:
            old_edge = next((e for e in ref.edges_out if e.end is next_node), None)
            if old_edge:
                self.removeEdge(old_edge)

        # Connect ref -> node -> next_node
        self.addNode(node)
        self.addEdge(Edge(start=ref, end=node, **kwargs))

        if next_node is not None:
            self.addEdge(Edge(start=node, end=next_node, **kwargs))
        else:
            # ref was the tail -> node is the new tail
            self._tail = node

    def insertAfterByKey(self, ref_key: str, new_key: str, **kwargs):
        """
        Inserts a new node built from 'new_key' after the node with 'ref_key'.
        """
        ref = self.getNodeByKey(ref_key)
        if ref is None:
            raise ValueError(f'No node found with key "{ref_key}".')
        self.insertAfter(ref, Node(new_key), **kwargs)

    def insertBefore(self, ref: Node, node: Node, **kwargs):
        """
        Inserts 'node' immediately before 'ref'.

        Before: prev_node -> ref
        After:  prev_node -> node -> ref

        Parameters:
            ref  (Node): Reference node already in the list.
            node (Node): Node to insert. Must not already be in the list.

        Raises:
            ValueError: If 'ref' is not in the list or 'node' is already in it.
        """
        if ref.key not in self.nodes:
            raise ValueError(f'Reference node "{ref.key}" is not in the list.')
        if node.key in self.nodes:
            raise ValueError(f'Node "{node.key}" is already in the list.')

        prev_node = self.getPrev(ref)

        if prev_node is None:
            # ref is the head,  prepend
            self.prepend(node, **kwargs)
        else:
            self.insertAfter(prev_node, node, **kwargs)

    def insertBeforeByKey(self, ref_key: str, new_key: str, **kwargs):
        """
        Inserts a new node built from 'new_key' before the node with 'ref_key'.
        """
        ref = self.getNodeByKey(ref_key)
        if ref is None:
            raise ValueError(f'No node found with key "{ref_key}".')
        self.insertBefore(ref, Node(new_key), **kwargs)

    def remove(self, node: Node):
        """
        Removes a node from the list, reconnecting its predecessor to its successor.

        Before: prev -> node -> next
        After:  prev -> next

        Parameters:
            node (Node): Node to remove.

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

    def removeByKey(self, key: str):
        """
        Removes the node with the given key from the list.
        """
        node = self.getNodeByKey(key)
        if node is None:
            raise ValueError(f'No node found with key "{key}".')
        self.remove(node)

    # endregion

    # region Is

    def isHead(self, node: Node) -> bool:
        """Returns True if 'node' is the head of the list."""
        return node is self._root

    def isTail(self, node: Node) -> bool:
        """Returns True if 'node' is the tail of the list."""
        return node is self._tail

    def isEmpty(self) -> bool:
        """Returns True if the list contains no nodes."""
        return len(self.nodes) == 0

    # endregion

    # region Get

    def getNext(self, node: Node) -> Node | None:
        """
        Returns the successor of 'node', or None if it is the tail.

        Returns:
            Node | None
        """
        for edge in node.edges_out:
            return edge.end
        return None

    def getPrev(self, node: Node) -> Node | None:
        """
        Returns the predecessor of 'node', or None if it is the head.

        Returns:
            Node | None
        """
        return self.getParent(node)

    def getNodeAt(self, index: int) -> Node | None:
        """
        Returns the node at position 'index' (0-based from the head).
        Returns None if the index is out of range.

        Parameters:
            index (int): Zero-based position.

        Returns:
            Node | None
        """
        if index < 0:
            raise ValueError('Index must be >= 0.')

        current = self.head
        for _ in range(index):
            if current is None:
                return None
            current = self.getNext(current)

        return current

    def getINodes(self, start: Node = None):
        """
        Iterates over all nodes from 'start' (default: head) to the tail.

        Yields:
            Node
        """
        current = start or self.head
        while current is not None:
            yield current
            current = self.getNext(current)

    def getNodes(self, start: Node = None) -> list[Node]:
        """
        Returns the ordered list of nodes from 'start' to the tail.

        Returns:
            list[Node]
        """
        return list(self.getINodes(start))

    # endregion

    # region Cast

    def toList(self) -> list:
        """
        Returns the list of node keys in order from head to tail.

        Returns:
            list[str]
        """
        return [node.key for node in self.getINodes()]

    def toDict(self) -> dict:
        """
        Extends the Tree dict representation with head and tail keys.

        Returns:
            dict
        """
        base = super().toDict()
        base['head'] = self.head.key if self.head else None
        base['tail'] = self.tail.key if self.tail else None
        return base

    # endregion

    # region Overwriting
    def __str__(self) -> str:
        head_key = self.head.key if self.head else 'None'
        tail_key = self.tail.key if self.tail else 'None'
        chain = ' -> '.join(self.toList()) if self.head else '(empty)'
        return (
            f"--- LinkedList '{self.name}' ---\n"
            f"Head  : {head_key}  |  Tail : {tail_key}\n"
            f"Length: {len(self)}\n"
            f"Chain : {chain}"
        )
    # endregion
