"""Tests for LinkedList.append."""

import pytest
from py_libraries.graph.Node import Node


def test_appends_to_an_empty_list(empty_list):
    node = Node('A')

    empty_list.append(node)

    assert empty_list.head is node
    assert empty_list.tail is node
    assert len(empty_list) == 1


def test_appends_after_the_tail(linked_list):
    node = Node('D')

    linked_list.append(node)

    assert linked_list.toList() == ['A', 'B', 'C', 'D']
    assert linked_list.tail is node


def test_raises_when_the_node_is_already_in_the_list(linked_list):
    with pytest.raises(ValueError, match='already in the list'):
        linked_list.append(linked_list.nodes['A'])


def test_forwards_the_edge_kwargs(linked_list):
    node = Node('D')

    linked_list.append(node, weight=7, color='blue')

    edge = next(e for e in linked_list.nodes['C'].edges_out)

    assert edge.weight == 7
    assert edge.color == 'blue'
