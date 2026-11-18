"""Tests for the LinkedList.head property."""

import pytest
from py_libraries.graph.LinkedList import LinkedList
from py_libraries.graph.Node import Node


def test_head_is_none_for_an_empty_list():
    assert LinkedList().head is None


def test_head_is_the_root(linked_list):
    assert linked_list.head is linked_list.root
    assert linked_list.head.key == 'A'


def test_setting_the_head_updates_the_root(linked_list):
    new_head = Node('H')

    linked_list.head = new_head

    assert linked_list.head is new_head
    assert linked_list.root is new_head
    assert new_head in linked_list


def test_setting_the_head_raises_when_the_node_has_parents(linked_list):
    with pytest.raises(ValueError, match="can't be a root"):
        linked_list.head = linked_list.nodes['B']
