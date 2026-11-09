"""Unit tests for containsKhmerChar."""

import pytest
from py_libraries.type.string import containsKhmerChar


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('abc', False),
        ('', False),
        ('123 .,;!', False),
        # KHMER LETTER KA
        ('\u1780', True),
        # KHMER SIGN ATTHACAN
        ('\u17dd', True),
        # KHMER DIGIT ZERO
        ('\u17e0', True),
        # KHMER SYMBOL PATHAMASAT
        ('\u19e0', True),
        # TAGALOG LETTER A: Tagalog
        ('\u1700', False),
        # MYANMAR LETTER KA: Myanmar
        ('\u1000', False),
    ],
)
def test_containsKhmerChar_known_values(s: str, expected: bool) -> None:
    assert containsKhmerChar(s) is expected


@pytest.mark.parametrize(
    'char',
    [
        '\u1780',
        '\u17dd',
        '\u17e0',
        '\u19e0',
    ],
)
def test_containsKhmerChar_char_inside_text_without_letters(char: str) -> None:
    assert containsKhmerChar(f'123 .,; {char} !? 456') is True
