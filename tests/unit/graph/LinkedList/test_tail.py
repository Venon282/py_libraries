"""Tests for the LinkedList.tail property."""

from py_libraries.graph.LinkedList import LinkedList
from py_libraries.graph.Node import Node


def test_tail_is_none_for_an_empty_list():
    assert LinkedList().tail is None


def test_tail_is_the_last_node(linked_list):
    assert linked_list.tail.key == 'C'
    assert linked_list.tail is linked_list.nodes['C']


def test_setting_the_tail_updates_the_tail(linked_list):
    new_tail = Node('T')

    linked_list.tail = new_tail

    assert linked_list.tail is new_tail


def test_tail_is_none_after_removing_the_last_node(linked_list):
    linked_list.remove(linked_list.nodes['C'])

    assert linked_list.tail.key == 'B'
