"""Unit tests for computeDigitalRoot."""

import pytest

from py_libraries.type.number import computeDigitalRoot


def repeatedDigitSum(num: int) -> int:
    """Reference implementation: sum the digits until only one is left.

    Args:
        num: Non-negative integer.

    Returns:
        The digital root of `num`.
    """
    while num >= 10:
        num = sum(int(digit) for digit in str(num))
    return num


@pytest.mark.parametrize(
    ('num', 'expected'),
    [
        (0, 0),
        (5, 5),
        (9, 9),
        (10, 1),
        (18, 9),
        (38, 2),
        (493193, 2),
        (999999999, 9),
    ],
)
def test_known_values(num: int, expected: int) -> None:
    """Return the single digit reached by summing the digits.

    Args:
        num: Number to reduce.
        expected: Expected digital root.
    """
    assert computeDigitalRoot(num) == expected


def test_matches_the_repeated_digit_sum() -> None:
    """Agree with the definition on a wide range of numbers."""
    for num in range(5000):
        assert computeDigitalRoot(num) == repeatedDigitSum(num)


def test_only_zero_has_a_zero_root() -> None:
    """Return 0 for zero only, and a digit from 1 to 9 for any positive number."""
    assert all(1 <= computeDigitalRoot(num) <= 9 for num in range(1, 1000))
