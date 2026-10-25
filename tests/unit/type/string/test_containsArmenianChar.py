"""Unit tests for containsArmenianChar."""

import pytest
from py_libraries.type.string import containsArmenianChar


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('abc', False),
        ('', False),
        ('123 .,;!', False),
        # ARMENIAN CAPITAL LETTER AYB
        ('\u0531', True),
        # ARMENIAN CAPITAL LETTER FEH
        ('\u0556', True),
        # ARMENIAN SMALL LETTER AYB
        ('\u0561', True),
        # ARMENIAN SMALL LIGATURE ECH YIWN
        ('\u0587', True),
        # ARMENIAN FULL STOP
        ('\u0589', True),
        # ARMENIAN DRAM SIGN
        ('\u058f', True),
        # HEBREW LETTER ALEF: Hebrew
        ('\u05d0', False),
        # GEORGIAN LETTER AN: Georgian
        ('\u10d0', False),
    ],
)
def test_containsArmenianChar_known_values(s: str, expected: bool) -> None:
    assert containsArmenianChar(s) is expected


@pytest.mark.parametrize(
    'char',
    [
        '\u0531',
        '\u0556',
        '\u0561',
        '\u0587',
        '\u0589',
        '\u058f',
    ],
)
def test_containsArmenianChar_char_inside_text_without_letters(char: str) -> None:
    assert containsArmenianChar(f'123 .,; {char} !? 456') is True
