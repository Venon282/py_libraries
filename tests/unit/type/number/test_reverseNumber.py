"""Unit tests for reverseNumber."""

import pytest

from py_libraries.type.number import reverseNumber


@pytest.mark.parametrize(
    ('n', 'expected'),
    [
        (123, 321),
        (-123, -321),
        (120, 21),
        (1000, 1),
        (-1200, -21),
        (0, 0),
        (5, 5),
        (-5, -5),
        (1221, 1221),
    ],
)
def test_known_values(n: int, expected: int) -> None:
    """Reverse the digits, keep the sign and drop the trailing zeros.

    Args:
        n: Number to reverse.
        expected: Expected result.
    """
    assert reverseNumber(n) == expected


@pytest.mark.parametrize('n', [1, 12, 907, 123456789, 10**15 + 7])
def test_reversing_twice_restores_a_number_without_trailing_zero(n: int) -> None:
    """Give back the number when the digits are reversed twice.

    Args:
        n: Positive number that does not end with a zero.
    """
    assert reverseNumber(reverseNumber(n)) == n


@pytest.mark.parametrize('n', [1, 120, 907, 123456789])
def test_negating_the_input_negates_the_result(n: int) -> None:
    """Treat a negative number like its opposite, with the sign restored.

    Args:
        n: Positive number to reverse.
    """
    assert reverseNumber(-n) == -reverseNumber(n)
