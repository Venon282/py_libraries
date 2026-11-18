"""Tests for LinkedList.getNext."""


def test_returns_the_successor(linked_list):
    assert linked_list.getNext(linked_list.nodes['A']) is linked_list.nodes['B']
    assert linked_list.getNext(linked_list.nodes['B']) is linked_list.nodes['C']


def test_returns_none_for_the_tail(linked_list):
    assert linked_list.getNext(linked_list.nodes['C']) is None
