"""Tests for LinkedList.insertAfterByKey."""

import pytest


def test_inserts_after_a_middle_node(linked_list):
    linked_list.insertAfterByKey('A', 'X')

    assert linked_list.toList() == ['A', 'X', 'B', 'C']


def test_inserts_after_the_tail_and_updates_the_tail(linked_list):
    linked_list.insertAfterByKey('C', 'D')

    assert linked_list.toList() == ['A', 'B', 'C', 'D']
    assert linked_list.tail.key == 'D'


def test_raises_when_the_reference_key_is_unknown(linked_list):
    with pytest.raises(ValueError, match='No node found'):
        linked_list.insertAfterByKey('Z', 'X')
