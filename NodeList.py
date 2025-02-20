from Node import Node
from Edge import Edge


class ListNode(Node):
    def __init__(self, value=0, prev_edge=None, next_edge=None, **kwargs):
        """
        Initializes a ListNode with inherited graphical and positional properties.
        The node's data is stored in 'value'. Its list links (previous/next) are represented
        by edges (max two: one incoming and one outgoing).
        """
        edges = []
        if prev_edge is not None:
            edges.append(prev_edge)
            self.head = False
        else:
            self.head = True

        if next_edge is not None:
            edges.append(next_edge)
            self.feet = False
        else:
            self.feet = True

        self.prev_edge = prev_edge
        self.next_edge = next_edge

        kwargs.setdefault('edges', edges)
        kwargs.setdefault('value', value)
        super().__init__(**kwargs)

    def setPrevEdge(self, prev_edge):
        """Alternative method to set the previous edge directly."""
        self.removePrev()
        self.prev_edge = prev_edge
        self.edges.append(prev_edge)
        self.head = False

    def setNextEdge(self, next_edge):
        """Alternative method to set the next edge directly."""
        self.removeNext()
        self.next_edge = next_edge
        self.edges.append(next_edge)
        self.feet = False

    @property
    def next(self):
        """
        Returns the next ListNode (i.e. the node connected by an outgoing edge),
        or None if no such edge exists.
        """
        if self.feet or self.next_edge is None:
            return None
        return self.next_edge.end

    @next.setter
    def next(self, next_node):
        """
        Sets the next pointer of this node by creating (or replacing) the outgoing edge.

        Parameters:
            next_node (ListNode): The node to be set as next.
        """
        self.removeNext()  # Remove any existing outgoing edge
        edge = Edge(start=self, end=next_node)
        self.edges.append(edge)
        if edge not in next_node.edges:
            next_node.edges.append(edge)
        self.next_edge = edge
        self.feet = False

    @next.deleter
    def next(self):
        """
        Deletes the next pointer by removing the outgoing edge.
        """
        self.removeNext()

    def removeNext(self):
        """
        Removes the outgoing edge representing the next pointer, if it exists.
        """
        to_remove = None
        for edge in self.edges:
            if edge.start == self:
                to_remove = edge
                break
        if to_remove:
            self.edges.remove(to_remove)
            if to_remove in to_remove.end.edges:
                to_remove.end.edges.remove(to_remove)
            self.next_edge = None
            self.feet = True

    @property
    def prev(self):
        """
        Returns the previous ListNode (i.e. the node connected by an incoming edge),
        or None if no such edge exists.
        """
        if self.head or self.prev_edge is None:
            return None
        return self.prev_edge.start

    @prev.setter
    def prev(self, prev_node):
        """
        Sets the previous pointer of this node by creating (or replacing) an incoming edge.

        Parameters:
            prev_node (ListNode): The node to be set as previous.
        """
        self.removePrev()  # Remove any existing incoming edge
        edge = Edge(start=prev_node, end=self)
        self.edges.append(edge)
        if edge not in prev_node.edges:
            prev_node.edges.append(edge)
        self.prev_edge = edge
        self.head = False

    @prev.deleter
    def prev(self):
        """
        Deletes the previous pointer by removing the incoming edge.
        """
        self.removePrev()

    def removePrev(self):
        """
        Removes the incoming edge representing the previous pointer, if it exists.
        """
        to_remove = None
        for edge in self.edges:
            if edge.end == self and edge.start != self:
                to_remove = edge
                break
        if to_remove:
            self.edges.remove(to_remove)
            if to_remove in to_remove.start.edges:
                to_remove.start.edges.remove(to_remove)
            self.prev_edge = None
            self.head = True

    def __str__(self):
        """
        Returns a string representation of the ListNode showing its value,
        along with its previous and next node values.
        """
        next_val = self.next.value if self.next is not None else "None"
        prev_val = self.prev.value if self.prev is not None else "None"
        return f"ListNode(value={self.value}, prev={prev_val}, next={next_val})"

    @staticmethod
    def mergeSortedLists(list1, list2):
        """
        Merges two sorted linked lists (of ListNode objects) and returns the head of the new sorted list.
        Uses a dummy head to simplify the merge process.
        """
        dummy = ListNode()  # Dummy node to start the new list
        tail = dummy

        # Merge nodes from both lists
        while list1 is not None and list2 is not None:
            if list1.value < list2.value:
                tail.next = ListNode(list1.value)
                list1 = list1.next
            else:
                tail.next = ListNode(list2.value)
                list2 = list2.next
            tail = tail.next

        # Append the remaining nodes from the non-empty list
        remaining = list1 if list1 is not None else list2
        while remaining is not None:
            tail.next = ListNode(remaining.value)
            tail = tail.next
            remaining = remaining.next

        # The dummy node's next pointer is the head of the merged list
        return dummy.next
