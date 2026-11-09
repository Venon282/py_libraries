"""Unit tests for intToRoman."""

import pytest

from py_libraries.type.number import intToRoman, romanToInt


@pytest.mark.parametrize(
    ('num', 'expected'),
    [
        (1, 'I'),
        (3, 'III'),
        (4, 'IV'),
        (9, 'IX'),
        (14, 'XIV'),
        (40, 'XL'),
        (58, 'LVIII'),
        (90, 'XC'),
        (400, 'CD'),
        (900, 'CM'),
        (1994, 'MCMXCIV'),
        (3999, 'MMMCMXCIX'),
    ],
)
def test_known_values(num: int, expected: str) -> None:
    """Write the numbers in subtractive notation.

    Args:
        num: Integer to convert.
        expected: Expected Roman numeral.
    """
    assert intToRoman(num) == expected


def test_round_trip_with_romanToInt_on_the_whole_standard_range() -> None:
    """Give back every number from 1 to 3999 once converted and read again."""
    for num in range(1, 4000):
        assert romanToInt(intToRoman(num)) == num


@pytest.mark.parametrize('num', [0, -1, -58])
def test_zero_and_negative_numbers_give_an_empty_string(num: int) -> None:
    """Return an empty string when there is nothing to write.

    Args:
        num: Zero or negative integer.
    """
    assert intToRoman(num) == ''


def test_numbers_above_3999_repeat_the_largest_symbol() -> None:
    """Repeat `M` because the notation has no larger symbol."""
    assert intToRoman(4000) == 'MMMM'
    assert intToRoman(5003) == 'MMMMMIII'
