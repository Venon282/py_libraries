"""Tests for LinkedList.toList."""

from py_libraries.graph.LinkedList import LinkedList


def test_returns_the_keys_in_order(linked_list):
    assert linked_list.toList() == ['A', 'B', 'C']


def test_returns_an_empty_list_for_an_empty_list():
    assert LinkedList().toList() == []
