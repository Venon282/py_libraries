"""Unit tests for containsCyrillicChar."""

import pytest
from py_libraries.type.string import containsCyrillicChar


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('abc', False),
        ('', False),
        ('123 .,;!', False),
        # CYRILLIC CAPITAL LETTER A
        ('\u0410', True),
        # CYRILLIC SMALL LETTER A
        ('\u0430', True),
        # CYRILLIC CAPITAL LETTER IE WITH GRAVE
        ('\u0400', True),
        # CYRILLIC SMALL LETTER HA WITH STROKE
        ('\u04ff', True),
        # CYRILLIC CAPITAL LETTER KOMI DE
        ('\u0500', True),
        # GREEK CAPITAL REVERSED DOTTED LUNATE SIGMA SYMBOL: Greek
        ('\u03ff', False),
        # ARMENIAN CAPITAL LETTER AYB: Armenian
        ('\u0531', False),
    ],
)
def test_containsCyrillicChar_known_values(s: str, expected: bool) -> None:
    assert containsCyrillicChar(s) is expected


@pytest.mark.parametrize(
    'char',
    [
        '\u0410',
        '\u0430',
        '\u0400',
        '\u04ff',
        '\u0500',
    ],
)
def test_containsCyrillicChar_char_inside_text_without_letters(char: str) -> None:
    assert containsCyrillicChar(f'123 .,; {char} !? 456') is True
