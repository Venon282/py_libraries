"""Unit tests for containsGeorgianChar."""

import pytest
from py_libraries.type.string import containsGeorgianChar


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('abc', False),
        ('', False),
        ('123 .,;!', False),
        # GEORGIAN LETTER AN
        ('\u10d0', True),
        # GEORGIAN CAPITAL LETTER AN
        ('\u10a0', True),
        # GEORGIAN LETTER LABIAL SIGN
        ('\u10ff', True),
        # GEORGIAN MTAVRULI CAPITAL LETTER AN
        ('\u1c90', True),
        # GEORGIAN SMALL LETTER AN
        ('\u2d00', True),
        # GEORGIAN PARAGRAPH SEPARATOR: no script (Common)
        ('\u10fb', False),
        # ARMENIAN CAPITAL LETTER AYB: Armenian
        ('\u0531', False),
    ],
)
def test_containsGeorgianChar_known_values(s: str, expected: bool) -> None:
    assert containsGeorgianChar(s) is expected


@pytest.mark.parametrize(
    'char',
    [
        '\u10d0',
        '\u10a0',
        '\u10ff',
        '\u1c90',
        '\u2d00',
    ],
)
def test_containsGeorgianChar_char_inside_text_without_letters(char: str) -> None:
    assert containsGeorgianChar(f'123 .,; {char} !? 456') is True
