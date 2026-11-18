"""Tests for LinkedList.toDict."""

from py_libraries.graph.LinkedList import LinkedList


def test_extends_the_tree_dict_with_head_and_tail(linked_list):
    result = linked_list.toDict()

    assert set(result) == {'nodes', 'edges', 'root', 'head', 'tail'}
    assert result['head'] == 'A'
    assert result['tail'] == 'C'
    assert result['root'] == 'A'
    assert len(result['nodes']) == 3
    assert len(result['edges']) == 2


def test_head_and_tail_are_none_for_an_empty_list():
    result = LinkedList().toDict()

    assert result['head'] is None
    assert result['tail'] is None
    assert result['root'] is None
