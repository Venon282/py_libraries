"""Unit tests for circularMean."""

import numpy as np
import pytest

from py_libraries.type.array import circularMean


def test_ring_values_are_averaged() -> None:
    """Replace the pixels of a ring by their mean, around the central pixel."""
    arr = np.array([[1, 0, 3], [0, 7, 0], [5, 0, 7]])

    result = circularMean(arr)

    np.testing.assert_allclose(result, [[4, 0, 4], [0, 7, 0], [4, 0, 4]])


def test_explicit_center_defines_the_rings() -> None:
    """Group the pixels by their distance to the given center."""
    arr = np.array([[1, 2], [3, 4]])

    result = circularMean(arr, center=(0, 0))

    np.testing.assert_allclose(result, [[1, 2.5], [2.5, 4]])


def test_large_threshold_puts_every_pixel_in_one_ring() -> None:
    """Give the mean of the whole array when a single ring holds every pixel."""
    arr = np.arange(20, dtype=float).reshape(4, 5)

    result = circularMean(arr, threshold=100.0)

    np.testing.assert_allclose(result, np.full(arr.shape, arr.mean()))


@pytest.mark.parametrize('shape', [(5, 5), (4, 7), (1, 9), (9, 1)])
def test_constant_array_is_unchanged(shape: tuple[int, int]) -> None:
    """Return the same constant, whatever the shape.

    Args:
        shape: Shape of the array.
    """
    arr = np.full(shape, 2.5)

    np.testing.assert_allclose(circularMean(arr), arr)


@pytest.mark.parametrize('seed', range(3))
def test_total_sum_is_preserved(seed: int) -> None:
    """Keep the sum of the pixels, because each ring is replaced by its mean.

    Args:
        seed: Seed of the random generator that fills the array.
    """
    arr = np.random.default_rng(seed).normal(size=(11, 8))

    result = circularMean(arr, center=(3, 2))

    assert result.sum() == pytest.approx(arr.sum())


def test_pixels_at_the_same_distance_share_the_same_value() -> None:
    """Give one value to the four pixels at the same distance on the axes."""
    arr = np.random.default_rng(0).normal(size=(9, 9))

    result = circularMean(arr)

    for k in range(1, 5):
        values = [
            result[4, 4 + k],
            result[4, 4 - k],
            result[4 + k, 4],
            result[4 - k, 4],
        ]
        assert values == pytest.approx([values[0]] * 4)


def test_integer_input_gives_a_float_result_and_is_not_modified() -> None:
    """Return floats and leave the input untouched."""
    arr = np.arange(9).reshape(3, 3)
    original = arr.copy()

    result = circularMean(arr)

    assert result.dtype == np.float64
    assert result.shape == arr.shape
    np.testing.assert_array_equal(arr, original)


@pytest.mark.filterwarnings('error')
@pytest.mark.parametrize('shape', [(1, 12), (12, 1), (2, 15)])
def test_unused_ring_indices_raise_no_warning(shape: tuple[int, int]) -> None:
    """Skip the ring indices without pixel instead of dividing by zero.

    Args:
        shape: Shape of a narrow array, whose distances leave gaps between rings.
    """
    arr = np.arange(shape[0] * shape[1], dtype=float).reshape(shape)

    result = circularMean(arr)

    assert np.isfinite(result).all()


@pytest.mark.parametrize('shape', [(5,), (2, 3, 4), ()])
def test_non_2d_array_raises_value_error(shape: tuple[int, ...]) -> None:
    """Raise ValueError when the array is not 2D.

    Args:
        shape: Shape of the invalid array.
    """
    with pytest.raises(ValueError):
        circularMean(np.zeros(shape))
