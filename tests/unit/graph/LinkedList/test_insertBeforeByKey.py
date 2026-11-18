"""Tests for LinkedList.insertBeforeByKey."""

import pytest


def test_inserts_before_a_middle_node(linked_list):
    linked_list.insertBeforeByKey('B', 'X')

    assert linked_list.toList() == ['A', 'X', 'B', 'C']


def test_inserts_before_the_head_and_updates_the_head(linked_list):
    linked_list.insertBeforeByKey('A', 'Z')

    assert linked_list.toList() == ['Z', 'A', 'B', 'C']
    assert linked_list.head.key == 'Z'


def test_raises_when_the_reference_key_is_unknown(linked_list):
    with pytest.raises(ValueError, match='No node found'):
        linked_list.insertBeforeByKey('Z', 'X')
