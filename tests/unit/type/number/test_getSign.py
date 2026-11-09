"""Unit tests for getSign."""

import pytest

from py_libraries.type.number import getSign


@pytest.mark.parametrize(
    ('n', 'expected'),
    [
        (5, 1),
        (1, 1),
        (-3, -1),
        (-1, -1),
        (0, 0),
        (0.5, 1),
        (-0.5, -1),
        (0.0, 0),
        (-0.0, 0),
        (float('inf'), 1),
        (float('-inf'), -1),
        (10**30, 1),
        (-(10**30), -1),
    ],
)
def test_known_values(n: float, expected: int) -> None:
    """Return 1, -1 or 0 according to the sign.

    Args:
        n: Number to inspect.
        expected: Expected sign.
    """
    assert getSign(n) == expected


def test_nan_gives_zero() -> None:
    """Return 0 for NaN, which is neither positive nor negative."""
    assert getSign(float('nan')) == 0


@pytest.mark.parametrize('n', [3, -3, 0, 0.25, -0.25])
def test_result_is_an_int(n: float) -> None:
    """Return a builtin int, not a bool.

    Args:
        n: Number to inspect.
    """
    assert type(getSign(n)) is int


@pytest.mark.parametrize('n', [3, 0.25, 10**9])
def test_sign_of_the_opposite_is_the_opposite_sign(n: float) -> None:
    """Flip the sign when the number is negated.

    Args:
        n: Positive number to inspect.
    """
    assert getSign(-n) == -getSign(n)
