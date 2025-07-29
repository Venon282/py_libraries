from Tree import Tree

class LinkedList(Tree):
    """
    A doubly linked list implemented as a subclass of Tree.

    Each node has attributes `val`, `next`, and `prev`. Internally it leverages Tree's
    directed edges for visual/graph operations but exposes a classic doubly‐linked interface.
    """
    class Node(Tree.root.__class__):
        def __init__(self, val=None):
            super().__init__(value=val)
            self.val = val
            self.next = None
            self.prev = None

    def __init__(self, iterable=None):
        """
        Initialize an empty LinkedList or build from an iterable of values.
        """
        super().__init__(root=None)
        self.head = None
        self.tail = None
        self._size = 0
        if iterable:
            for val in iterable:
                self.append(val)

    def append(self, val):
        """Append a new node with `val` at the end of the list."""
        new_node = LinkedList.Node(val)
        if not self.head:
            # Empty list
            self.head = self.tail = self.root = new_node
            self.addNode(new_node)
        else:
            # Link via Tree
            self.addChild(self.tail, new_node)
            # Classic pointers
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self._size += 1
        return new_node

    def prepend(self, val):
        """Insert a new node with `val` at the beginning of the list."""
        new_node = LinkedList.Node(val)
        if not self.head:
            self.head = self.tail = self.root = new_node
            self.addNode(new_node)
        else:
            self.addChild(new_node, self.head)
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
            self.root = new_node
        self._size += 1
        return new_node

    def find(self, val):
        """Return the first node with matching `val`, else None."""
        curr = self.head
        while curr:
            if curr.val == val:
                return curr
            curr = curr.next
        return None

    def insert_after(self, target_val, val):
        """Insert a new node with `val` after the first node with `target_val`."""
        target = self.find(target_val)
        if not target:
            raise ValueError(f"Value {target_val} not found")
        new_node = LinkedList.Node(val)
        successor = target.next
        # Tree edges
        self.addChild(target, new_node)
        if successor:
            old_edge = next(e for e in self.edges if e.start is target and e.end is successor)
            self.removeEdge(old_edge)
            self.addChild(new_node, successor)
        # Classic pointers
        new_node.prev = target
        new_node.next = successor
        target.next = new_node
        if successor:
            successor.prev = new_node
        else:
            # New tail
            self.tail = new_node
        self._size += 1
        return new_node

    def remove(self, val):
        """Remove the first occurrence of `val` and return its node."""
        curr = self.find(val)
        if not curr:
            raise ValueError(f"Value {val} not found")
        predecessor = curr.prev
        successor = curr.next
        # Unlink Tree edges
        if predecessor:
            e1 = next(e for e in self.edges if e.start is predecessor and e.end is curr)
            self.removeEdge(e1)
        if successor:
            e2 = next(e for e in self.edges if e.start is curr and e.end is successor)
            self.removeEdge(e2)
        if predecessor and successor:
            self.addChild(predecessor, successor)
        # Classic pointers
        if predecessor:
            predecessor.next = successor
        if successor:
            successor.prev = predecessor
        if curr is self.head:
            self.head = successor
            self.root = successor
        if curr is self.tail:
            self.tail = predecessor
        # Clean up
        curr.next = curr.prev = None
        self.removeNode(curr)
        self._size -= 1
        return curr

    def to_list(self, reverse=False):
        """Return Python list of values in forward or reverse order."""
        result, curr = [], self.tail if reverse else self.head
        while curr:
            result.append(curr.val)
            curr = curr.prev if reverse else curr.next
        return result

    def __iter__(self):
        curr = self.head
        while curr:
            yield curr.val
            curr = curr.next

    def __reversed__(self):
        curr = self.tail
        while curr:
            yield curr.val
            curr = curr.prev

    def __len__(self):
        return self._size

    def is_valid_list(self):
        """Validate doubly-linked integrity: correct sizes, no cycles, pointers match edges."""
        if self._size == 0:
            return self.head is None and self.tail is None and not self.edges
        # Check forward chain
        count, curr = 0, self.head
        prev_node = None
        while curr:
            count += 1
            if curr.prev is not prev_node:
                return False
            prev_node = curr
            curr = curr.next
        if count != self._size:
            return False
        # Check backward chain
        count, curr = 0, self.tail
        next_node = None
        while curr:
            count += 1
            if curr.next is not next_node:
                return False
            next_node = curr
            curr = curr.prev
        if count != self._size:
            return False
        # Edge count should match size-1
        return len(self.edges) == self._size - 1

    def __str__(self):
        return "DoublyLinkedList([" + ", ".join(str(v) for v in self.to_list()) + "])"

    @staticmethod
    def is_maxima(node):
        """A maxima is b as a < b > c."""
        return node.prev.val < node.val > node.next.val if node.prev and node.next else False
    
    @staticmethod
    def is_minima(node):
        """A minima is b as a > b < c."""
        return node.prev.val > node.val < node.next.val if node.prev and node.next else False
    
    @staticmethod
    def is_critical_node(node):
        """A critical point is b as a > b < c."""
        return ((node.prev.val < node.val > node.next.val) or (node.prev.val > node.val < node.next.val)) if node.prev and node.next else False
    
    @staticmethod
    def critical_points(head):
        critical_points = []
        node = head.next
        while node:
            if LinkedList.is_critical_node(node):
                critical_points.append(node)
            node = node.next
        return critical_points
    
    @staticmethod
    def critical_points_positions(head):
        critical_points_positions = []
        node = head.next
        i = 1
        while node:
            if LinkedList.is_critical_node(node):
                critical_points_positions.append(i)
            node = node.next
            i+=1
        return critical_points_positions