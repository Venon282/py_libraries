"""Unit tests for circularTrimmedMean."""

import math

import numpy as np
import pytest

from py_libraries.type.array import circularMean, circularTrimmedMean


def bruteForce(
    arr: np.ndarray,
    center: tuple[int, int],
    threshold: float,
    proportion: float,
) -> np.ndarray:
    """Reference implementation: sort and slice the values of each ring.

    Args:
        arr: Input 2D array.
        center: Center coordinates (y, x).
        threshold: Radius increment between two consecutive rings.
        proportion: Proportion to trim from each end of a ring.

    Returns:
        Array where each pixel is replaced by the trimmed mean of its ring.
    """
    rings: dict[int, list[float]] = {}
    pixel_ring: dict[tuple[int, int], int] = {}
    for (y, x), value in np.ndenumerate(arr):
        ring = round(math.hypot(x - center[1], y - center[0]) / threshold)
        rings.setdefault(ring, []).append(float(value))
        pixel_ring[(y, x)] = ring

    means = {}
    for ring, values in rings.items():
        cut = int(proportion * len(values))
        kept = sorted(values)[cut : len(values) - cut]
        means[ring] = sum(kept) / len(kept)

    result = np.empty(arr.shape)
    for (y, x), ring in pixel_ring.items():
        result[y, x] = means[ring]
    return result


def test_ring_extremes_are_trimmed() -> None:
    """Drop the lowest and the highest value of a four pixels ring."""
    arr = np.array([[1, 0, 2], [0, 5, 0], [3, 0, 100]])

    result = circularTrimmedMean(arr, proportiontocut=0.25)

    expected = [[2.5, 0.0, 2.5], [0.0, 5.0, 0.0], [2.5, 0.0, 2.5]]
    np.testing.assert_allclose(result, expected)


def test_zero_proportion_is_the_plain_mean() -> None:
    """Give the same result as circularMean when nothing is trimmed."""
    arr = np.random.default_rng(0).normal(size=(8, 9))

    np.testing.assert_allclose(
        circularTrimmedMean(arr, proportiontocut=0.0), circularMean(arr)
    )


def test_default_proportion_does_not_trim_rings_below_ten_pixels() -> None:
    """Leave the rings of fewer than ten pixels untrimmed with the default 0.1."""
    arr = np.random.default_rng(1).normal(size=(3, 3))

    np.testing.assert_allclose(circularTrimmedMean(arr), circularMean(arr))


def test_explicit_center_defines_the_rings() -> None:
    """Group the pixels by their distance to the given center."""
    arr = np.array([[1, 2], [3, 4]])

    result = circularTrimmedMean(arr, center=(0, 0))

    np.testing.assert_allclose(result, [[1, 2.5], [2.5, 4]])


@pytest.mark.parametrize('proportion', [0.1, 0.2, 0.4])
def test_matches_brute_force(proportion: float) -> None:
    """Agree with a sort and slice implementation on a random array.

    Args:
        proportion: Proportion trimmed from each end of a ring.
    """
    arr = np.random.default_rng(2).normal(size=(15, 13))

    result = circularTrimmedMean(arr, (7, 6), 0.9, proportion)

    np.testing.assert_allclose(result, bruteForce(arr, (7, 6), 0.9, proportion))


@pytest.mark.parametrize('shape', [(5, 5), (1, 9)])
def test_constant_array_is_unchanged(shape: tuple[int, int]) -> None:
    """Return the same constant, whatever the shape.

    Args:
        shape: Shape of the array.
    """
    arr = np.full(shape, 3.0)

    np.testing.assert_allclose(circularTrimmedMean(arr), arr)


def test_proportion_too_big_raises_value_error() -> None:
    """Raise ValueError when scipy rejects the proportion for a large ring."""
    arr = np.random.default_rng(0).normal(size=(9, 9))

    with pytest.raises(ValueError):
        circularTrimmedMean(arr, proportiontocut=0.9)


@pytest.mark.parametrize('shape', [(5,), (2, 3, 4), ()])
def test_non_2d_array_raises_value_error(shape: tuple[int, ...]) -> None:
    """Raise ValueError when the array is not 2D.

    Args:
        shape: Shape of the invalid array.
    """
    with pytest.raises(ValueError):
        circularTrimmedMean(np.zeros(shape))
