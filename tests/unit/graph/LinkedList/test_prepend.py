"""Tests for LinkedList.prepend."""

import pytest
from py_libraries.graph.Node import Node


def test_prepends_to_an_empty_list(empty_list):
    node = Node('A')

    empty_list.prepend(node)

    assert empty_list.head is node
    assert empty_list.tail is node
    assert len(empty_list) == 1


def test_prepends_before_the_head(linked_list):
    node = Node('Z')

    linked_list.prepend(node)

    assert linked_list.toList() == ['Z', 'A', 'B', 'C']
    assert linked_list.head is node
    assert linked_list.tail.key == 'C'


def test_raises_when_the_node_is_already_in_the_list(linked_list):
    with pytest.raises(ValueError, match='already in the list'):
        linked_list.prepend(linked_list.nodes['A'])


def test_forwards_the_edge_kwargs(linked_list):
    node = Node('Z')

    linked_list.prepend(node, weight=9, color='purple')

    edge = next(e for e in node.edges_out)

    assert edge.weight == 9
    assert edge.color == 'purple'
