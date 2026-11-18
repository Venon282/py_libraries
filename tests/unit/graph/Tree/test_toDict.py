"""Tests for Tree.toDict."""

from py_libraries.graph.Tree import Tree


def test_extends_the_graph_dict_with_the_root(tree):
    result = tree.toDict()

    assert set(result) == {'nodes', 'edges', 'root'}
    assert result['root'] == 'R'
    assert len(result['nodes']) == 4
    assert len(result['edges']) == 3


def test_root_is_none_when_there_is_no_root():
    assert Tree().toDict()['root'] is None
