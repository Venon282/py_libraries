"""Shared fixtures for the graph.Tree tests."""

import pytest
from py_libraries.graph.Node import Node
from py_libraries.graph.Tree import Tree


@pytest.fixture
def tree() -> Tree:
    """A tree with root R, children A and B, and C as a child of A."""
    result = Tree('t')
    r = Node('R', x=0.0, y=0.0)
    a = Node('A', x=1.0, y=1.0)
    b = Node('B', x=2.0, y=1.0)
    c = Node('C', x=3.0, y=2.0)
    result.root = r
    result.addChild(r, a)
    result.addChild(r, b)
    result.addChild(a, c)
    return result
