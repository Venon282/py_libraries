"""Unit tests for containsIndianChar."""

import pytest
from py_libraries.type.string import containsIndianChar


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('abc', False),
        ('', False),
        ('123 .,;!', False),
        # DEVANAGARI LETTER A
        ('\u0905', True),
        # BENGALI LETTER A
        ('\u0985', True),
        # GURMUKHI LETTER A
        ('\u0a05', True),
        # GUJARATI LETTER A
        ('\u0a85', True),
        # ORIYA LETTER A
        ('\u0b05', True),
        # TAMIL LETTER A
        ('\u0b85', True),
        # TELUGU LETTER A
        ('\u0c05', True),
        # KANNADA LETTER A
        ('\u0c85', True),
        # MALAYALAM LETTER A
        ('\u0d05', True),
        # OL CHIKI LETTER LA
        ('\u1c5a', True),
        # MEETEI MAYEK LETTER KOK
        ('\uabc0', True),
        # SINHALA LETTER AYANNA: Sinhala
        ('\u0d85', False),
        # TIBETAN LETTER KA: Tibetan
        ('\u0f40', False),
        # THAI CHARACTER KO KAI: Thai
        ('\u0e01', False),
        # DEVANAGARI DANDA: danda (Common)
        ('\u0964', False),
        # ARABIC LETTER ALEF: Arabic
        ('\u0627', False),
    ],
)
def test_containsIndianChar_known_values(s: str, expected: bool) -> None:
    assert containsIndianChar(s) is expected


@pytest.mark.parametrize(
    'char',
    [
        '\u0905',
        '\u0985',
        '\u0a05',
        '\u0a85',
        '\u0b05',
        '\u0b85',
        '\u0c05',
        '\u0c85',
        '\u0d05',
        '\u1c5a',
        '\uabc0',
    ],
)
def test_containsIndianChar_char_inside_text_without_letters(char: str) -> None:
    assert containsIndianChar(f'123 .,; {char} !? 456') is True
