"""Unit tests for containsLatinChar."""

import pytest
from py_libraries.type.string import containsLatinChar


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('abc', True),
        ('Hello World', True),
        ('', False),
        ('123 .,;!', False),
        # LATIN SMALL LETTER A
        ('\u0061', True),
        # LATIN CAPITAL LETTER Z
        ('\u005a', True),
        # LATIN SMALL LETTER E WITH ACUTE
        ('\u00e9', True),
        # FEMININE ORDINAL INDICATOR
        ('\u00aa', True),
        # LATIN CAPITAL LETTER A WITH MACRON
        ('\u0100', True),
        # LATIN SMALL LETTER Y WITH STROKE
        ('\u024f', True),
        # FULLWIDTH LATIN CAPITAL LETTER A
        ('\uff21', True),
        # MULTIPLICATION SIGN: no script (Common)
        ('\u00d7', False),
        # DIVISION SIGN: no script (Common)
        ('\u00f7', False),
        # GREEK SMALL LETTER ALPHA: Greek
        ('\u03b1', False),
        # CYRILLIC CAPITAL LETTER ZHE: Cyrillic
        ('\u0416', False),
        # CJK UNIFIED IDEOGRAPH-4E00: Han
        ('\u4e00', False),
        # MATHEMATICAL BOLD CAPITAL A: letter of no script (Common)
        ('\U0001d400', False),
        # COMBINING ACUTE ACCENT: combining mark (Inherited)
        ('\u0301', False),
        # GRINNING FACE: emoji (Common)
        ('\U0001f600', False),
    ],
)
def test_containsLatinChar_known_values(s: str, expected: bool) -> None:
    assert containsLatinChar(s) is expected


@pytest.mark.parametrize(
    'char',
    [
        '\u0061',
        '\u005a',
        '\u00e9',
        '\u00aa',
        '\u0100',
        '\u024f',
        '\uff21',
    ],
)
def test_containsLatinChar_char_inside_text_without_letters(char: str) -> None:
    assert containsLatinChar(f'123 .,; {char} !? 456') is True
