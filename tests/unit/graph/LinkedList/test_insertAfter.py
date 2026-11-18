"""Tests for LinkedList.insertAfter."""

import pytest
from py_libraries.graph.Node import Node


def test_inserts_after_a_middle_node(linked_list):
    node = Node('X')

    linked_list.insertAfter(linked_list.nodes['A'], node)

    assert linked_list.toList() == ['A', 'X', 'B', 'C']
    assert linked_list.head.key == 'A'
    assert linked_list.tail.key == 'C'


def test_inserts_after_the_tail_and_updates_the_tail(linked_list):
    node = Node('D')

    linked_list.insertAfter(linked_list.nodes['C'], node)

    assert linked_list.toList() == ['A', 'B', 'C', 'D']
    assert linked_list.tail is node


def test_raises_when_the_reference_is_not_in_the_list(linked_list):
    with pytest.raises(ValueError, match='not in the list'):
        linked_list.insertAfter(Node('Z'), Node('X'))


def test_raises_when_the_node_is_already_in_the_list(linked_list):
    with pytest.raises(ValueError, match='already in the list'):
        linked_list.insertAfter(linked_list.nodes['A'], linked_list.nodes['B'])


def test_forwards_the_edge_kwargs(linked_list):
    node = Node('X')

    linked_list.insertAfter(linked_list.nodes['A'], node, weight=6, color='red')

    edges = {e.end.key: e for e in linked_list.edges}

    assert edges['X'].weight == 6
    assert edges['B'].weight == 6
    assert edges['X'].color == 'red'
    assert edges['B'].color == 'red'
