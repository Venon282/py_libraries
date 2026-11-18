"""Tests for LinkedList.__init__."""

from py_libraries.graph.LinkedList import LinkedList


def test_empty_list_has_no_head_or_tail():
    linked_list = LinkedList()

    assert linked_list.head is None
    assert linked_list.tail is None
    assert len(linked_list) == 0


def test_default_name_is_undefined():
    assert LinkedList().name == 'undefined'


def test_name_can_be_set():
    assert LinkedList('my_list').name == 'my_list'


def test_is_directed():
    assert LinkedList().is_directed is True
