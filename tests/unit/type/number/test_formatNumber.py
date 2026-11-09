"""Unit tests for formatNumber."""

from typing import cast

import numpy as np
import pytest

from py_libraries.type.number import formatNumber


@pytest.mark.parametrize(
    ('num', 'expected'),
    [
        (0, '0'),
        (0.0, '0'),
        (-0.0, '0'),
        (1.0, '1'),
        (1.5, '1.5'),
        (100, '100'),
        (-2.5, '-2.5'),
        (0.654212574854, '0.654'),
        (123456.789, '123456.789'),
        (0.001, '0.001'),
        (0.0000000256, '2.56e-8'),
        (0.0009, '9e-4'),
        (-0.00012, '-1.2e-4'),
        (455351356, '4.554e8'),
        (1e6, '1e6'),
        (-1e9, '-1e9'),
        (1e100, '1e100'),
    ],
)
def test_known_values(num: float, expected: str) -> None:
    """Write fixed-point or scientific notation, without useless zeros.

    Args:
        num: Number to format.
        expected: Expected string.
    """
    assert formatNumber(num) == expected


@pytest.mark.parametrize(
    ('num', 'expected'),
    [('1e-7', '1e-7'), ('2.50', '2.5'), ('0', '0'), ('1234567', '1.235e6')],
)
def test_strings_are_converted_with_float(num: str, expected: str) -> None:
    """Format a string that holds a number.

    Args:
        num: String to convert and format.
        expected: Expected string.
    """
    assert formatNumber(num) == expected


def test_numpy_scalar_is_accepted() -> None:
    """Format a numpy scalar, which is not a builtin float."""
    assert formatNumber(np.float32(0.5)) == '0.5'


@pytest.mark.parametrize('num', ['abc', '', '1,5'])
def test_unconvertible_string_is_returned_unchanged(num: str) -> None:
    """Return the string itself when `float` rejects it.

    Args:
        num: String that is not a number.
    """
    assert formatNumber(num) == num


def test_unconvertible_object_is_returned_as_a_string() -> None:
    """Return `str(num)` when `float` raises a TypeError."""
    assert formatNumber(cast(float, None)) == 'None'
    assert formatNumber(cast(float, [1, 2])) == '[1, 2]'


@pytest.mark.parametrize(
    ('num', 'expected'),
    [(float('inf'), 'inf'), (float('-inf'), '-inf'), (float('nan'), 'nan')],
)
def test_non_finite_values(num: float, expected: str) -> None:
    """Return `inf`, `-inf` and `nan` as such.

    Args:
        num: Non-finite number.
        expected: Expected string.
    """
    assert formatNumber(num) == expected


@pytest.mark.parametrize(
    ('num', 'bound_min', 'bound_max', 'precision', 'expected'),
    [
        (1500, 0.001, 1000, 3, '1.5e3'),
        (999, 0.001, 1000, 3, '999'),
        (1000, 0.001, 1000, 3, '1e3'),
        (0.002, 0.001, 1e6, 3, '0.002'),
        (0.002, 0.01, 1e6, 3, '2e-3'),
        (0.00002, 1e-5, 1e6, 6, '0.00002'),
        (3.14159, 0.001, 1e6, 2, '3.14'),
        (3.7, 0.001, 1e6, 0, '4'),
        (123456789, 0.001, 1e6, 1, '1.2e8'),
        (123456789, 0.001, 1e6, 0, '1e8'),
    ],
)
def test_custom_bounds_and_precision(
    num: float,
    bound_min: float,
    bound_max: float,
    precision: int,
    expected: str,
) -> None:
    """Apply the bounds of the fixed-point notation and the precision.

    Args:
        num: Number to format.
        bound_min: Lower bound of the fixed-point notation.
        bound_max: Upper bound of the fixed-point notation.
        precision: Maximum number of decimals.
        expected: Expected string.
    """
    assert formatNumber(num, bound_min, bound_max, precision) == expected


def test_value_below_the_resolution_is_rounded_to_zero() -> None:
    """Write `0` for a fixed-point value smaller than the last decimal."""
    assert formatNumber(0.0002, bound_min=1e-5, precision=3) == '0'


def test_fixed_point_values_keep_the_requested_absolute_precision() -> None:
    """Stay within half a unit of the last decimal in the fixed-point range."""
    rng = np.random.default_rng(0)

    for value in rng.uniform(0.001, 1e6, size=500):
        result = float(formatNumber(float(value)))

        assert result == pytest.approx(value, abs=5e-4 + 1e-9)


def test_scientific_values_keep_the_requested_relative_precision() -> None:
    """Stay within half a unit of the last mantissa decimal out of the range."""
    rng = np.random.default_rng(1)
    magnitudes = 10.0 ** rng.uniform(-30, -3.1, size=250)
    magnitudes = np.concatenate([magnitudes, 10.0 ** rng.uniform(6, 30, size=250)])

    for value in magnitudes:
        result = float(formatNumber(float(value)))

        assert result == pytest.approx(value, rel=5e-4 + 1e-9)
