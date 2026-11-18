"""Tests for LinkedList.addChild."""

import pytest
from py_libraries.graph.Node import Node


def test_adds_a_single_child_to_the_tail(linked_list):
    # Only the tail has no child yet, so it is the only valid parent.
    child = Node('X')

    linked_list.addChild(linked_list.nodes['C'], child)

    assert linked_list.getNext(linked_list.nodes['C']) is child


def test_raises_when_the_parent_already_has_a_child(linked_list):
    with pytest.raises(ValueError, match='already has a child'):
        linked_list.addChild(linked_list.nodes['A'], Node('X'))


def test_updates_the_tail_when_the_child_is_appended_at_the_end(linked_list):
    new_tail = Node('D')

    linked_list.addChild(linked_list.nodes['C'], new_tail)

    assert linked_list.tail is new_tail


def test_stores_the_edge_properties(linked_list):
    child = Node('X')

    linked_list.addChild(
        linked_list.nodes['C'],
        child,
        weight=5,
        color='red',
        width=2,
        properties={'tag': 1},
    )

    edge = next(e for e in linked_list.nodes['C'].edges_out)

    assert edge.weight == 5
    assert edge.color == 'red'
    assert edge.width == 2
    assert edge.properties == {'tag': 1}
