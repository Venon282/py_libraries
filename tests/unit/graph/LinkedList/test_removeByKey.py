"""Tests for LinkedList.removeByKey."""

import pytest


def test_removes_a_middle_node(linked_list):
    linked_list.removeByKey('B')

    assert linked_list.toList() == ['A', 'C']


def test_removes_the_head(linked_list):
    linked_list.removeByKey('A')

    assert linked_list.toList() == ['B', 'C']
    assert linked_list.head.key == 'B'


def test_removes_the_tail(linked_list):
    linked_list.removeByKey('C')

    assert linked_list.toList() == ['A', 'B']
    assert linked_list.tail.key == 'B'


def test_raises_when_the_key_is_unknown(linked_list):
    with pytest.raises(ValueError, match='No node found'):
        linked_list.removeByKey('Z')
