"""Tests for LinkedList.isEmpty."""

from py_libraries.graph.LinkedList import LinkedList


def test_an_empty_list_is_empty():
    assert LinkedList().isEmpty() is True


def test_a_list_with_nodes_is_not_empty(linked_list):
    assert linked_list.isEmpty() is False
