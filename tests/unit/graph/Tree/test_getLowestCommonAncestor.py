"""Tests for Tree.getLowestCommonAncestor."""

from py_libraries.graph.Node import Node


def test_returns_the_closest_common_ancestor(tree):
    assert (
        tree.getLowestCommonAncestor(tree.nodes['C'], tree.nodes['B'])
        is tree.nodes['R']
    )


def test_returns_none_when_there_is_no_common_ancestor(tree):
    orphan = Node('Z')
    tree.addNode(orphan)

    assert tree.getLowestCommonAncestor(tree.nodes['C'], orphan) is None


def test_returns_the_root_when_one_node_is_the_ancestor_of_the_other(tree):
    # A is the ancestor of C, but A's own ancestors are only [R], so the
    # closest common ancestor found is R.
    assert (
        tree.getLowestCommonAncestor(tree.nodes['A'], tree.nodes['C'])
        is tree.nodes['R']
    )
