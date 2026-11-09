"""Unit tests for circularBilateral."""

import math

import numpy as np
import pytest

from py_libraries.type.array import circularBilateral, circularMean


def bruteForce(
    arr: np.ndarray,
    center: tuple[int, int],
    threshold: float,
    sigma_i: float,
) -> np.ndarray:
    """Reference implementation: weight the values of each ring one by one.

    Args:
        arr: Input 2D array.
        center: Center coordinates (y, x).
        threshold: Radius increment between two consecutive rings.
        sigma_i: Intensity sigma of the filter.

    Returns:
        Array where each pixel is replaced by the weighted mean of its ring.
    """
    rings: dict[int, list[float]] = {}
    pixel_ring: dict[tuple[int, int], int] = {}
    for (y, x), value in np.ndenumerate(arr):
        ring = round(math.hypot(x - center[1], y - center[0]) / threshold)
        rings.setdefault(ring, []).append(float(value))
        pixel_ring[(y, x)] = ring

    filtered = {}
    for ring, values in rings.items():
        mean = sum(values) / len(values)
        weights = [math.exp(-((v - mean) ** 2) / (2 * sigma_i**2)) for v in values]
        filtered[ring] = sum(w * v for w, v in zip(weights, values, strict=True)) / sum(
            weights
        )

    result = np.empty(arr.shape)
    for (y, x), ring in pixel_ring.items():
        result[y, x] = filtered[ring]
    return result


def test_outlier_of_a_ring_is_rejected() -> None:
    """Ignore the value that is far from the mean of its ring."""
    arr = np.array([[1.0, 0, 1], [0, 5, 0], [1, 0, 100]])

    result = circularBilateral(arr, sigma_i=1.0)

    corners = [result[0, 0], result[0, 2], result[2, 0], result[2, 2]]
    assert corners == pytest.approx([1.0] * 4)


def test_large_sigma_is_the_plain_mean() -> None:
    """Give the same result as circularMean when every weight is about 1."""
    arr = np.random.default_rng(0).uniform(size=(9, 8))

    np.testing.assert_allclose(
        circularBilateral(arr, sigma_i=1e6), circularMean(arr), rtol=1e-9
    )


@pytest.mark.parametrize('shape', [(5, 5), (4, 7), (1, 9)])
def test_constant_array_is_unchanged(shape: tuple[int, int]) -> None:
    """Return the same constant, whatever the shape.

    Args:
        shape: Shape of the array.
    """
    arr = np.full(shape, 4.0)

    np.testing.assert_allclose(circularBilateral(arr), arr)


@pytest.mark.parametrize('sigma_i', [0.05, 0.3, 2.0])
def test_matches_brute_force(sigma_i: float) -> None:
    """Agree with a pixel by pixel implementation on a random array.

    Args:
        sigma_i: Intensity sigma of the filter.
    """
    arr = np.random.default_rng(3).normal(scale=0.3, size=(9, 11))

    result = circularBilateral(arr, center=(4, 5), threshold=0.8, sigma_i=sigma_i)

    np.testing.assert_allclose(result, bruteForce(arr, (4, 5), 0.8, sigma_i))


def test_result_stays_within_the_range_of_the_input() -> None:
    """Keep every output value between the minimum and the maximum of the input."""
    arr = np.random.default_rng(0).normal(size=(10, 10))

    result = circularBilateral(arr, sigma_i=0.5)

    assert np.min(arr) <= np.min(result)
    assert np.max(result) <= np.max(arr)


def test_explicit_center_defines_the_rings() -> None:
    """Group the pixels by their distance to the given center."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = circularBilateral(arr, center=(0, 0), sigma_i=1e6)

    np.testing.assert_allclose(result, [[1, 2.5], [2.5, 4]])


@pytest.mark.parametrize('shape', [(5,), (2, 3, 4), ()])
def test_non_2d_array_raises_value_error(shape: tuple[int, ...]) -> None:
    """Raise ValueError when the array is not 2D.

    Args:
        shape: Shape of the invalid array.
    """
    with pytest.raises(ValueError):
        circularBilateral(np.zeros(shape))
