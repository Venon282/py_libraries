"""Tests for LinkedList.getPrev."""


def test_returns_the_predecessor(linked_list):
    assert linked_list.getPrev(linked_list.nodes['B']) is linked_list.nodes['A']
    assert linked_list.getPrev(linked_list.nodes['C']) is linked_list.nodes['B']


def test_returns_none_for_the_head(linked_list):
    assert linked_list.getPrev(linked_list.nodes['A']) is None
