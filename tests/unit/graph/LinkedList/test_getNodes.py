"""Tests for LinkedList.getNodes."""

from py_libraries.graph.LinkedList import LinkedList


def test_returns_the_nodes_from_the_head_to_the_tail(linked_list):
    assert linked_list.getNodes() == [
        linked_list.nodes['A'],
        linked_list.nodes['B'],
        linked_list.nodes['C'],
    ]


def test_returns_the_nodes_from_a_start_node(linked_list):
    assert linked_list.getNodes(linked_list.nodes['B']) == [
        linked_list.nodes['B'],
        linked_list.nodes['C'],
    ]


def test_returns_an_empty_list_for_an_empty_list():
    assert LinkedList().getNodes() == []
