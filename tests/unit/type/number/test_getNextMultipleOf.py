"""Unit tests for getNextMultipleOf."""

import numpy as np
import pytest

from py_libraries.type.number import getNextMultipleOf


@pytest.mark.parametrize(
    ('n', 'multiple', 'expected'),
    [
        (7, 3, 9),
        (9, 3, 9),
        (0, 5, 0),
        (1, 1, 1),
        (1, 8, 8),
        (10, 10, 10),
        (11, 10, 20),
        (-7, 3, -6),
        (-6, 3, -6),
    ],
)
def test_known_values(n: int, multiple: int, expected: int) -> None:
    """Return the smallest multiple that is not below the number.

    Args:
        n: Number to align.
        multiple: Multiple to align to.
        expected: Expected result.
    """
    assert getNextMultipleOf(n, multiple) == expected


@pytest.mark.parametrize('seed', range(5))
def test_result_is_the_closest_multiple_not_below_n(seed: int) -> None:
    """Return a multiple of `multiple` in the range [n, n + multiple).

    Args:
        seed: Seed of the random generator that picks the inputs.
    """
    rng = np.random.default_rng(seed)

    for _ in range(100):
        n = int(rng.integers(-1000, 1000))
        multiple = int(rng.integers(1, 50))

        result = getNextMultipleOf(n, multiple)

        assert result % multiple == 0
        assert n <= result < n + multiple


def test_zero_multiple_raises_zero_division_error() -> None:
    """Raise ZeroDivisionError when the multiple is 0."""
    with pytest.raises(ZeroDivisionError):
        getNextMultipleOf(5, 0)
