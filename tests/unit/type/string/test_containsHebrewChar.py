"""Unit tests for containsHebrewChar."""

import pytest
from py_libraries.type.string import containsHebrewChar


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('abc', False),
        ('', False),
        ('123 .,;!', False),
        # HEBREW LETTER ALEF
        ('\u05d0', True),
        # HEBREW LETTER TAV
        ('\u05ea', True),
        # HEBREW LIGATURE YIDDISH DOUBLE VAV
        ('\u05f0', True),
        # HEBREW LETTER SHIN WITH SHIN DOT
        ('\ufb2a', True),
        # ARABIC LETTER ALEF: Arabic
        ('\u0627', False),
        # GREEK SMALL LETTER ALPHA: Greek
        ('\u03b1', False),
    ],
)
def test_containsHebrewChar_known_values(s: str, expected: bool) -> None:
    assert containsHebrewChar(s) is expected


@pytest.mark.parametrize(
    'char',
    [
        '\u05d0',
        '\u05ea',
        '\u05f0',
        '\ufb2a',
    ],
)
def test_containsHebrewChar_char_inside_text_without_letters(char: str) -> None:
    assert containsHebrewChar(f'123 .,; {char} !? 456') is True
