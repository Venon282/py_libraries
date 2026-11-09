"""Unit tests for containsGreekChar."""

import pytest
from py_libraries.type.string import containsGreekChar


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('abc', False),
        ('', False),
        ('123 .,;!', False),
        # GREEK SMALL LETTER ALPHA
        ('\u03b1', True),
        # GREEK CAPITAL LETTER ALPHA
        ('\u0391', True),
        # GREEK CAPITAL LETTER HETA
        ('\u0370', True),
        # GREEK CAPITAL REVERSED DOTTED LUNATE SIGMA SYMBOL
        ('\u03ff', True),
        # GREEK SMALL LETTER ALPHA WITH PSILI
        ('\u1f00', True),
        # CYRILLIC CAPITAL LETTER IE WITH GRAVE: Cyrillic
        ('\u0400', False),
        # COMBINING LATIN SMALL LETTER X: combining mark (Inherited)
        ('\u036f', False),
        # MICRO SIGN: no script (Common)
        ('\u00b5', False),
        # CJK UNIFIED IDEOGRAPH-4E00: Han
        ('\u4e00', False),
    ],
)
def test_containsGreekChar_known_values(s: str, expected: bool) -> None:
    assert containsGreekChar(s) is expected


@pytest.mark.parametrize(
    'char',
    [
        '\u03b1',
        '\u0391',
        '\u0370',
        '\u03ff',
        '\u1f00',
    ],
)
def test_containsGreekChar_char_inside_text_without_letters(char: str) -> None:
    assert containsGreekChar(f'123 .,; {char} !? 456') is True
