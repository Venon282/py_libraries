"""Tests for LinkedList.insertBefore."""

import pytest
from py_libraries.graph.Node import Node


def test_inserts_before_a_middle_node(linked_list):
    node = Node('X')

    linked_list.insertBefore(linked_list.nodes['B'], node)

    assert linked_list.toList() == ['A', 'X', 'B', 'C']


def test_inserts_before_the_head_and_updates_the_head(linked_list):
    node = Node('Z')

    linked_list.insertBefore(linked_list.nodes['A'], node)

    assert linked_list.toList() == ['Z', 'A', 'B', 'C']
    assert linked_list.head is node
    assert linked_list.tail.key == 'C'


def test_raises_when_the_reference_is_not_in_the_list(linked_list):
    with pytest.raises(ValueError, match='not in the list'):
        linked_list.insertBefore(Node('Z'), Node('X'))


def test_raises_when_the_node_is_already_in_the_list(linked_list):
    with pytest.raises(ValueError, match='already in the list'):
        linked_list.insertBefore(linked_list.nodes['B'], linked_list.nodes['A'])


def test_forwards_the_edge_kwargs(linked_list):
    node = Node('X')

    linked_list.insertBefore(linked_list.nodes['B'], node, weight=2, color='teal')

    edges = {e.end.key: e for e in linked_list.edges}

    assert edges['X'].weight == 2
    assert edges['B'].weight == 2
    assert edges['X'].color == 'teal'
    assert edges['B'].color == 'teal'
