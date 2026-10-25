"""Unit tests for containsMongolianChar."""

import pytest
from py_libraries.type.string import containsMongolianChar


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('abc', False),
        ('', False),
        ('123 .,;!', False),
        # MONGOLIAN LETTER A
        ('\u1820', True),
        # MONGOLIAN LETTER CHI
        ('\u1842', True),
        # MONGOLIAN LETTER MANCHU ALI GALI LHA
        ('\u18aa', True),
        # CANADIAN SYLLABICS HYPHEN: Canadian Aboriginal
        ('\u1400', False),
        # CYRILLIC CAPITAL LETTER A: Cyrillic
        ('\u0410', False),
    ],
)
def test_containsMongolianChar_known_values(s: str, expected: bool) -> None:
    assert containsMongolianChar(s) is expected


@pytest.mark.parametrize(
    'char',
    [
        '\u1820',
        '\u1842',
        '\u18aa',
    ],
)
def test_containsMongolianChar_char_inside_text_without_letters(char: str) -> None:
    assert containsMongolianChar(f'123 .,; {char} !? 456') is True
