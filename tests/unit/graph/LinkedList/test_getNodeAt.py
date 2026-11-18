"""Tests for LinkedList.getNodeAt."""

import pytest
from py_libraries.graph.LinkedList import LinkedList


def test_returns_the_node_at_the_given_index(linked_list):
    assert linked_list.getNodeAt(0) is linked_list.nodes['A']
    assert linked_list.getNodeAt(1) is linked_list.nodes['B']
    assert linked_list.getNodeAt(2) is linked_list.nodes['C']


def test_returns_none_when_the_index_is_out_of_range(linked_list):
    assert linked_list.getNodeAt(3) is None
    assert linked_list.getNodeAt(10) is None


def test_returns_none_for_an_empty_list():
    assert LinkedList().getNodeAt(0) is None


def test_raises_when_the_index_is_negative(linked_list):
    with pytest.raises(ValueError, match='Index must be >= 0'):
        linked_list.getNodeAt(-1)
