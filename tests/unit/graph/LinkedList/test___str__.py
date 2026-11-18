"""Tests for LinkedList.__str__."""

from py_libraries.graph.LinkedList import LinkedList


def test_str_contains_the_summary(linked_list):
    text = str(linked_list)

    assert "--- LinkedList 'undefined' ---" in text
    assert 'Head  : A  |  Tail : C' in text
    assert 'Length: 3' in text
    assert 'Chain : A -> B -> C' in text


def test_str_shows_empty_when_there_is_no_head():
    text = str(LinkedList())

    assert 'Head  : None  |  Tail : None' in text
    assert 'Length: 0' in text
    assert 'Chain : (empty)' in text


def test_str_uses_the_name(linked_list):
    linked_list.name = 'my_list'

    assert "--- LinkedList 'my_list' ---" in str(linked_list)
