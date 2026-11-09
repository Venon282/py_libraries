"""Unit tests for romanToInt."""

import pytest

from py_libraries.type.number import romanToInt


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('I', 1),
        ('V', 5),
        ('X', 10),
        ('L', 50),
        ('C', 100),
        ('D', 500),
        ('M', 1000),
        ('III', 3),
        ('XII', 12),
        ('LVIII', 58),
        ('IV', 4),
        ('IX', 9),
        ('XL', 40),
        ('XC', 90),
        ('CD', 400),
        ('CM', 900),
        ('MCMXCIV', 1994),
        ('MMMCMXCIX', 3999),
    ],
)
def test_known_values(s: str, expected: int) -> None:
    """Convert the symbols and the subtractive pairs.

    Args:
        s: Roman numeral to convert.
        expected: Expected integer.
    """
    assert romanToInt(s) == expected


def test_empty_string_gives_zero() -> None:
    """Return 0 for an empty numeral."""
    assert romanToInt('') == 0


def test_malformed_numeral_is_converted_without_validation() -> None:
    """Convert a subtractive pair even when it is not a standard numeral."""
    assert romanToInt('IC') == 99


@pytest.mark.parametrize('s', ['a', 'iv', 'Z', 'XIA', 'AX', '1'])
def test_unknown_symbol_raises_key_error(s: str) -> None:
    """Raise KeyError for a character that is not an upper-case Roman symbol.

    Args:
        s: Numeral holding an unknown character.
    """
    with pytest.raises(KeyError):
        romanToInt(s)
