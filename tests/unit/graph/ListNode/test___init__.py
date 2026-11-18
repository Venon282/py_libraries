"""Tests for ListNode.__init__."""

from py_libraries.graph.ListNode import ListNode


def test_default_values():
    node = ListNode()

    assert node.val == 0
    assert node.next is None


def test_custom_value():
    assert ListNode(42).val == 42


def test_custom_next():
    next_node = ListNode(2)
    node = ListNode(1, next=next_node)

    assert node.val == 1
    assert node.next is next_node


def test_builds_a_chain():
    # 1 -> 2 -> 3
    head = ListNode(1, next=ListNode(2, next=ListNode(3)))

    assert head.val == 1
    assert head.next.val == 2
    assert head.next.next.val == 3
    assert head.next.next.next is None
