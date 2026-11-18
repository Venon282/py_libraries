"""Tests for LinkedList.prependByKey."""


def test_prepends_a_new_node_built_from_a_key(empty_list):
    empty_list.prependByKey('A')

    assert empty_list.toList() == ['A']
    assert empty_list.head.key == 'A'
    assert empty_list.tail.key == 'A'


def test_prepends_before_the_head(linked_list):
    linked_list.prependByKey('Z')

    assert linked_list.toList() == ['Z', 'A', 'B', 'C']
    assert linked_list.head.key == 'Z'
    assert linked_list.tail.key == 'C'


def test_forwards_the_edge_kwargs(linked_list):
    linked_list.prependByKey('Z', weight=4, color='orange')

    edge = next(e for e in linked_list.nodes['Z'].edges_out)

    assert edge.weight == 4
    assert edge.color == 'orange'
