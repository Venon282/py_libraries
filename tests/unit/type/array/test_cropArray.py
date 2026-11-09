"""Unit tests for cropArray."""

import numpy as np
import pytest

from py_libraries.type.array import cropArray


@pytest.mark.parametrize(
    ('new_size', 'method', 'expected'),
    [
        (4, 'left', [0, 1, 2, 3]),
        (4, 'right', [6, 7, 8, 9]),
        (4, 'middle', [3, 4, 5, 6]),
        (5, 'middle', [2, 3, 4, 5, 6]),
        (10, 'left', list(range(10))),
        (10, 'middle', list(range(10))),
        (0, 'left', []),
    ],
)
def test_one_dimension_known_values(
    new_size: int, method: str, expected: list[int]
) -> None:
    """Keep the requested part of a 1D array.

    Args:
        new_size: Target size.
        method: Where to crop from.
        expected: Expected values.
    """
    np.testing.assert_array_equal(cropArray(np.arange(10), new_size, method), expected)


@pytest.mark.parametrize(
    ('method', 'alias'), [('left', 'l'), ('right', 'r'), ('middle', 'm')]
)
def test_short_aliases_and_case_are_accepted(method: str, alias: str) -> None:
    """Accept the one-letter methods, in any case.

    Args:
        method: Full name of the method.
        alias: One-letter alias of the method.
    """
    arr = np.arange(10)
    expected = cropArray(arr, 4, method)

    np.testing.assert_array_equal(cropArray(arr, 4, alias), expected)
    np.testing.assert_array_equal(cropArray(arr, 4, alias.upper()), expected)
    np.testing.assert_array_equal(cropArray(arr, 4, method.upper()), expected)


def test_default_method_crops_from_the_left() -> None:
    """Crop from the beginning when no method is given."""
    np.testing.assert_array_equal(cropArray(np.arange(10), 3), [0, 1, 2])


def test_middle_removes_the_extra_element_at_the_end() -> None:
    """Remove one more element on the right when the removed count is odd."""
    np.testing.assert_array_equal(cropArray(np.arange(5), 2, 'middle'), [1, 2])


def test_int_size_only_crops_the_first_dimension() -> None:
    """Keep the other dimensions whole when the size is an int."""
    arr = np.arange(24).reshape(6, 4)

    result = cropArray(arr, 3, 'right')

    np.testing.assert_array_equal(result, arr[3:, :])


def test_tuple_size_crops_each_dimension() -> None:
    """Crop each of the first dimensions with its own size."""
    arr = np.arange(48).reshape(6, 8)

    result = cropArray(arr, (2, 4), 'middle')

    np.testing.assert_array_equal(result, arr[2:4, 2:6])


def test_tuple_shorter_than_the_array_keeps_the_last_dimensions_whole() -> None:
    """Leave the dimensions beyond the tuple untouched."""
    arr = np.arange(120).reshape(4, 5, 6)

    result = cropArray(arr, (2, 3), 'left')

    assert result.shape == (2, 3, 6)
    np.testing.assert_array_equal(result, arr[:2, :3, :])


@pytest.mark.parametrize('method', ['left', 'right', 'middle'])
def test_full_size_returns_the_same_values(method: str) -> None:
    """Return an equal array when the target size is the current size.

    Args:
        method: Where to crop from.
    """
    arr = np.arange(12).reshape(3, 4)

    np.testing.assert_array_equal(cropArray(arr, (3, 4), method), arr)


def test_result_is_a_view_of_the_input() -> None:
    """Share the memory of the input instead of copying it."""
    arr = np.arange(10)

    result = cropArray(arr, 4, 'middle')

    assert np.shares_memory(result, arr)


@pytest.mark.parametrize('seed', range(3))
def test_middle_matches_the_centered_slice(seed: int) -> None:
    """Remove `(n - size) // 2` elements on the left for any size.

    Args:
        seed: Seed of the random generator that picks the shape.
    """
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 40))
    size = int(rng.integers(0, n + 1))
    arr = rng.normal(size=n)

    start = (n - size) // 2

    np.testing.assert_array_equal(
        cropArray(arr, size, 'middle'), arr[start : start + size]
    )


def test_too_many_dimensions_raises_value_error() -> None:
    """Raise ValueError when the target has more dimensions than the array."""
    with pytest.raises(ValueError, match='cannot crop to 2 dimensions'):
        cropArray(np.arange(10), (2, 2))


def test_invalid_method_raises_value_error() -> None:
    """Raise ValueError for an unknown method."""
    with pytest.raises(ValueError, match='Invalid crop method'):
        cropArray(np.arange(10), 4, 'top')


@pytest.mark.parametrize(
    ('shape', 'new_size', 'dimension'),
    [((5,), 6, 0), ((4, 5), (4, 6), 1), ((4, 5), (5, 5), 0)],
)
def test_size_larger_than_the_dimension_raises_value_error(
    shape: tuple[int, ...], new_size: int | tuple[int, ...], dimension: int
) -> None:
    """Raise ValueError and name the dimension that is too small.

    Args:
        shape: Shape of the array.
        new_size: Target size.
        dimension: Index of the dimension that is too small.
    """
    with pytest.raises(ValueError, match=f'Cannot crop dimension {dimension}'):
        cropArray(np.zeros(shape), new_size)
