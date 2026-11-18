"""Tests for LinkedList.remove."""

import pytest
from py_libraries.graph.Node import Node


def test_removes_a_middle_node(linked_list):
    linked_list.remove(linked_list.nodes['B'])

    assert linked_list.toList() == ['A', 'C']
    assert linked_list.head.key == 'A'
    assert linked_list.tail.key == 'C'


def test_removes_the_head_and_updates_the_head(linked_list):
    linked_list.remove(linked_list.nodes['A'])

    assert linked_list.toList() == ['B', 'C']
    assert linked_list.head.key == 'B'
    assert linked_list.tail.key == 'C'


def test_removes_the_tail_and_updates_the_tail(linked_list):
    linked_list.remove(linked_list.nodes['C'])

    assert linked_list.toList() == ['A', 'B']
    assert linked_list.head.key == 'A'
    assert linked_list.tail.key == 'B'


def test_removes_the_only_node(linked_list):
    linked_list.remove(linked_list.nodes['A'])
    linked_list.remove(linked_list.nodes['B'])
    linked_list.remove(linked_list.nodes['C'])

    assert linked_list.toList() == []
    assert linked_list.head is None
    assert linked_list.tail is None


def test_raises_when_the_node_is_not_in_the_list(linked_list):
    with pytest.raises(ValueError, match='not in the list'):
        linked_list.remove(Node('Z'))
