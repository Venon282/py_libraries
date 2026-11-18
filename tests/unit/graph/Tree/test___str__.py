"""Tests for Tree.__str__."""

from py_libraries.graph.Tree import Tree


def test_str_contains_the_summary(tree):
    text = str(tree)

    assert "--- Tree 't' ---" in text
    assert 'Root    : R' in text
    assert 'Nodes   : 4  |  Edges : 3' in text
    assert 'Height  : 2  |  Leaves: 2' in text
    assert 'Valid   : True' in text


def test_str_shows_none_when_there_is_no_root():
    text = str(Tree())

    assert 'Root    : None' in text
    assert 'Valid   : False' in text
