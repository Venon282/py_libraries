"""Unit tests for circularMedian."""

import math

import numpy as np
import pytest

from py_libraries.type.array import circularMedian


def bruteForce(
    arr: np.ndarray, center: tuple[int, int], threshold: float
) -> np.ndarray:
    """Reference implementation: group the pixels one by one with plain Python.

    Args:
        arr: Input 2D array.
        center: Center coordinates (y, x).
        threshold: Radius increment between two consecutive rings.

    Returns:
        Array where each pixel is replaced by the median of its ring.
    """
    rings: dict[int, list[float]] = {}
    for (y, x), value in np.ndenumerate(arr):
        ring = round(math.hypot(x - center[1], y - center[0]) / threshold)
        rings.setdefault(ring, []).append(float(value))

    result = np.empty(arr.shape)
    for (y, x), _ in np.ndenumerate(arr):
        ring = round(math.hypot(x - center[1], y - center[0]) / threshold)
        result[y, x] = np.median(rings[ring])
    return result


def test_ring_values_are_replaced_by_their_median() -> None:
    """Replace the corners by their median, around the central pixel."""
    arr = np.array([[1, 0, 3], [0, 7, 0], [5, 0, 100]])

    result = circularMedian(arr)

    np.testing.assert_allclose(result, [[4, 0, 4], [0, 7, 0], [4, 0, 4]])


def test_outlier_does_not_move_the_median() -> None:
    """Ignore a single extreme value in a ring."""
    arr = np.array([[2, 0, 2], [0, 5, 0], [2, 0, 1000]])

    result = circularMedian(arr)

    assert result[0, 0] == result[0, 2] == result[2, 0] == result[2, 2] == 2


def test_explicit_center_defines_the_rings() -> None:
    """Group the pixels by their distance to the given center."""
    arr = np.array([[1, 2], [3, 4]])

    result = circularMedian(arr, center=(0, 0))

    np.testing.assert_allclose(result, [[1, 2.5], [2.5, 4]])


def test_large_threshold_puts_every_pixel_in_one_ring() -> None:
    """Give the median of the whole array when a single ring holds every pixel."""
    arr = np.arange(20, dtype=float).reshape(4, 5) ** 2

    result = circularMedian(arr, threshold=100.0)

    np.testing.assert_allclose(result, np.full(arr.shape, np.median(arr)))


@pytest.mark.parametrize('shape', [(5, 5), (4, 7), (1, 9)])
def test_constant_array_is_unchanged(shape: tuple[int, int]) -> None:
    """Return the same constant, whatever the shape.

    Args:
        shape: Shape of the array.
    """
    arr = np.full(shape, -1.5)

    np.testing.assert_allclose(circularMedian(arr), arr)


@pytest.mark.parametrize('seed', range(3))
def test_matches_brute_force(seed: int) -> None:
    """Agree with a pixel by pixel implementation on random arrays.

    Args:
        seed: Seed of the random generator that fills the array.
    """
    arr = np.random.default_rng(seed).normal(size=(9, 7))

    result = circularMedian(arr, center=(2, 3), threshold=0.9)

    np.testing.assert_allclose(result, bruteForce(arr, (2, 3), 0.9))


def test_result_stays_within_the_range_of_the_input() -> None:
    """Keep every output value between the minimum and the maximum of the input."""
    arr = np.random.default_rng(0).normal(size=(10, 10))

    result = circularMedian(arr)

    assert np.min(arr) <= np.min(result)
    assert np.max(result) <= np.max(arr)


def test_integer_input_gives_a_float_result_and_is_not_modified() -> None:
    """Return floats and leave the input untouched."""
    arr = np.arange(9).reshape(3, 3)
    original = arr.copy()

    result = circularMedian(arr)

    assert result.dtype == np.float64
    np.testing.assert_array_equal(arr, original)


@pytest.mark.parametrize('shape', [(5,), (2, 3, 4), ()])
def test_non_2d_array_raises_value_error(shape: tuple[int, ...]) -> None:
    """Raise ValueError when the array is not 2D.

    Args:
        shape: Shape of the invalid array.
    """
    with pytest.raises(ValueError):
        circularMedian(np.zeros(shape))
