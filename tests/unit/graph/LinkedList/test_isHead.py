"""Tests for LinkedList.isHead."""

from py_libraries.graph.Node import Node


def test_head_is_recognized(linked_list):
    assert linked_list.isHead(linked_list.nodes['A']) is True


def test_non_head_is_not_recognized(linked_list):
    assert linked_list.isHead(linked_list.nodes['B']) is False
    assert linked_list.isHead(linked_list.nodes['C']) is False


def test_a_node_outside_the_list_is_not_the_head(linked_list):
    assert linked_list.isHead(Node('Z')) is False
