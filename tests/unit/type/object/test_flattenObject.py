"""Unit tests for flattenObject."""

from typing import Any

import numpy as np
import pytest

from py_libraries.type.object import flattenObject


@pytest.mark.parametrize(
    'value',
    [3, 2.5, None, True, 'abc', '', b'ab', b'', bytearray(b'xy'), bytearray()],
)
def test_atomic_values_are_returned_as_a_single_item(value: Any) -> None:
    """Keep numbers, None, strings and bytes-like objects whole.

    Args:
        value: Value that must not be split.
    """
    assert flattenObject(value) == [value]


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ([], []),
        ((), []),
        ({}, []),
        (set(), []),
        (np.array([]), []),
        ([[], [[]], ()], []),
    ],
)
def test_empty_containers_flatten_to_an_empty_list(value: Any, expected: list) -> None:
    """Return an empty list for containers without any element.

    Args:
        value: Empty (possibly nested) container.
        expected: Expected flat list.
    """
    assert flattenObject(value) == expected


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ([1, 2, 3], [1, 2, 3]),
        ((1, (2, (3,))), [1, 2, 3]),
        ([1, [2, (3, [4])]], [1, 2, 3, 4]),
        ([[1, 2], [3, [4, [5]]]], [1, 2, 3, 4, 5]),
    ],
)
def test_nested_lists_and_tuples(value: Any, expected: list) -> None:
    """Flatten lists and tuples at any depth, keeping the order.

    Args:
        value: Nested lists and tuples.
        expected: Expected flat list.
    """
    assert flattenObject(value) == expected


def test_set_items_are_flattened() -> None:
    """Flatten the items of a set."""
    assert sorted(flattenObject({3, 1, 2})) == [1, 2, 3]


def test_strings_are_not_split_into_characters() -> None:
    """Keep each string inside nested containers as one item."""
    assert flattenObject(['ab', ['cd', ('ef',)]]) == ['ab', 'cd', 'ef']


def test_bytes_are_not_split_into_integers() -> None:
    """Keep each bytes and bytearray inside nested containers as one item."""
    value = [b'ab', [bytearray(b'cd')], ('e', b'f')]

    assert flattenObject(value) == [b'ab', bytearray(b'cd'), 'e', b'f']


def test_dict_is_flattened_by_values_and_keys_are_discarded() -> None:
    """Flatten a dict by its values only."""
    value = {'a': 1, 'b': [2, 3], 'c': {'d': 4}}

    assert flattenObject(value) == [1, 2, 3, 4]


def test_generator_is_consumed_and_flattened() -> None:
    """Flatten a generator, whose items can themselves be containers."""
    assert flattenObject(item for item in [1, [2, 3], (4,)]) == [1, 2, 3, 4]


@pytest.mark.parametrize('shape', [(6,), (2, 3), (2, 3, 4), (1, 1, 5, 2)])
def test_numpy_array_matches_ravel_in_row_major_order(shape: tuple[int, ...]) -> None:
    """Return the same items as `ndarray.ravel`.

    Args:
        shape: Shape of the random integer array.
    """
    rng = np.random.default_rng(0)
    array = rng.integers(0, 100, size=shape)

    assert flattenObject(array) == array.ravel().tolist()


def test_mixed_nested_structure() -> None:
    """Flatten arrays, dicts, tuples and atomic items in one structure."""
    value = [np.array([[1, 2], [3, 4]]), {'k': (5, 6)}, 'x', 7]

    assert flattenObject(value) == [1, 2, 3, 4, 5, 6, 'x', 7]


def test_flat_input_is_returned_unchanged_as_a_new_list() -> None:
    """Return a new list equal to an already flat list."""
    value = [1, 'a', None, 2.5]

    result = flattenObject(value)

    assert result == value
    assert result is not value
