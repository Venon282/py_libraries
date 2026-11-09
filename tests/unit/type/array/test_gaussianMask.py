"""Unit tests for gaussianMask."""

import numpy as np
import pytest

from py_libraries.type.array import gaussianMask


@pytest.mark.parametrize(('h', 'w'), [(1, 1), (3, 3), (4, 7), (8, 5), (16, 16)])
def test_shape_dtype_and_range(h: int, w: int) -> None:
    """Return a float32 array of shape (h, w) with values in (0, 1].

    Args:
        h: Height of the mask.
        w: Width of the mask.
    """
    mask = gaussianMask(h, w)

    assert mask.shape == (h, w)
    assert mask.dtype == np.float32
    assert mask.min() > 0
    assert mask.max() == 1.0


def test_known_values_of_a_three_by_three_mask() -> None:
    """Match the closed form of the gaussian at the center, edges and corners."""
    mask = gaussianMask(3, 3, sigma=0.4)

    expected = np.array(
        [
            [np.exp(-6.25), np.exp(-3.125), np.exp(-6.25)],
            [np.exp(-3.125), 1.0, np.exp(-3.125)],
            [np.exp(-6.25), np.exp(-3.125), np.exp(-6.25)],
        ]
    )
    np.testing.assert_allclose(mask, expected, rtol=1e-6)


@pytest.mark.parametrize(('h', 'w'), [(5, 5), (6, 9), (9, 6)])
def test_mask_is_symmetric(h: int, w: int) -> None:
    """Be unchanged by a flip along either axis.

    Args:
        h: Height of the mask.
        w: Width of the mask.
    """
    mask = gaussianMask(h, w)

    np.testing.assert_allclose(mask, np.flipud(mask))
    np.testing.assert_allclose(mask, np.fliplr(mask))


def test_mask_decreases_from_the_center_along_the_axes() -> None:
    """Fall monotonically from the center of the middle row to its end."""
    mask = gaussianMask(7, 9)

    row = mask[3, 4:]
    column = mask[3:, 4]

    assert np.all(np.diff(row) < 0)
    assert np.all(np.diff(column) < 0)


def test_larger_sigma_gives_a_flatter_mask() -> None:
    """Keep each pixel higher when sigma grows."""
    narrow = gaussianMask(9, 9, sigma=0.2)
    wide = gaussianMask(9, 9, sigma=0.8)

    assert np.all(wide >= narrow)
    assert wide.min() > narrow.min()


def test_square_mask_is_symmetric_by_transposition() -> None:
    """Be unchanged by a transposition when the mask is square."""
    mask = gaussianMask(8, 8)

    np.testing.assert_allclose(mask, mask.T)


@pytest.mark.parametrize(('h', 'w'), [(0, 5), (5, 0), (0, 0)])
def test_empty_mask_raises_value_error(h: int, w: int) -> None:
    """Raise ValueError because the maximum of an empty mask does not exist.

    Args:
        h: Height of the mask.
        w: Width of the mask.
    """
    with pytest.raises(ValueError):
        gaussianMask(h, w)
