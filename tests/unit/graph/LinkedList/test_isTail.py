"""Tests for LinkedList.isTail."""

from py_libraries.graph.Node import Node


def test_tail_is_recognized(linked_list):
    assert linked_list.isTail(linked_list.nodes['C']) is True


def test_non_tail_is_not_recognized(linked_list):
    assert linked_list.isTail(linked_list.nodes['A']) is False
    assert linked_list.isTail(linked_list.nodes['B']) is False


def test_a_node_outside_the_list_is_not_the_tail(linked_list):
    assert linked_list.isTail(Node('Z')) is False
