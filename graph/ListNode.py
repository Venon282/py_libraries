"""A standalone node of a singly linked list."""

from __future__ import annotations


class ListNode:
    """A node of a singly linked list.

    Unlike the graph-based `LinkedList`, this node is self-contained: it
    carries a value and a direct reference to the next node.

    Args:
        val: Value carried by the node.
        next: Node that follows this one in the list.
    """

    def __init__(
        self,
        val: int = 0,
        next: ListNode | None = None,
    ) -> None:
        self.val: int = val
        self.next: ListNode | None = next
