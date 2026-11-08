"""Unit tests for containsLaoChar."""

import pytest
from py_libraries.type.string import containsLaoChar


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('abc', False),
        ('', False),
        ('123 .,;!', False),
        # LAO LETTER KO
        ('\u0e81', True),
        # LAO LETTER O
        ('\u0ead', True),
        # LAO VOWEL SIGN E
        ('\u0ec0', True),
        # LAO LETTER KHMU NYO
        ('\u0edf', True),
        # THAI CHARACTER KO KAI: Thai
        ('\u0e01', False),
        # KHMER LETTER KA: Khmer
        ('\u1780', False),
    ],
)
def test_containsLaoChar_known_values(s: str, expected: bool) -> None:
    assert containsLaoChar(s) is expected


@pytest.mark.parametrize(
    'char',
    [
        '\u0e81',
        '\u0ead',
        '\u0ec0',
        '\u0edf',
    ],
)
def test_containsLaoChar_char_inside_text_without_letters(char: str) -> None:
    assert containsLaoChar(f'123 .,; {char} !? 456') is True
