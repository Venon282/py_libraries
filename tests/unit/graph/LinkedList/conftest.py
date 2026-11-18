"""Shared fixtures for the LinkedList tests."""

import pytest
from py_libraries.graph.LinkedList import LinkedList


def buildList(keys: list[str]) -> LinkedList:
    """Build a linked list from a sequence of keys.

    Args:
        keys: Keys of the nodes, in order from the head to the tail.

    Returns:
        The built linked list.
    """
    linked_list = LinkedList()

    for key in keys:
        linked_list.appendByKey(key)

    return linked_list


@pytest.fixture
def linked_list() -> LinkedList:
    """A linked list A -> B -> C."""
    return buildList(['A', 'B', 'C'])


@pytest.fixture
def empty_list() -> LinkedList:
    """An empty linked list."""
    return LinkedList()
