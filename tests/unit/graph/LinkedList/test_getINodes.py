"""Tests for LinkedList.getINodes."""

from py_libraries.graph.LinkedList import LinkedList


def test_iterates_from_the_head_to_the_tail(linked_list):
    keys = [node.key for node in linked_list.getINodes()]

    assert keys == ['A', 'B', 'C']


def test_iterates_from_a_start_node(linked_list):
    keys = [node.key for node in linked_list.getINodes(linked_list.nodes['B'])]

    assert keys == ['B', 'C']


def test_yields_nothing_for_an_empty_list():
    assert list(LinkedList().getINodes()) == []


def test_yields_nothing_when_the_start_is_none(linked_list):
    # Starting from the head (None) is the same as the default.
    keys = [node.key for node in linked_list.getINodes(None)]

    assert keys == ['A', 'B', 'C']
