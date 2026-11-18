"""Tests for LinkedList.appendByKey."""


def test_appends_a_new_node_built_from_a_key(empty_list):
    empty_list.appendByKey('A')

    assert empty_list.toList() == ['A']
    assert empty_list.head.key == 'A'
    assert empty_list.tail.key == 'A'


def test_appends_after_the_tail(linked_list):
    linked_list.appendByKey('D')

    assert linked_list.toList() == ['A', 'B', 'C', 'D']
    assert linked_list.tail.key == 'D'


def test_forwards_the_edge_kwargs(linked_list):
    linked_list.appendByKey('D', weight=3, color='green')

    edge = next(e for e in linked_list.nodes['C'].edges_out)

    assert edge.weight == 3
    assert edge.color == 'green'
