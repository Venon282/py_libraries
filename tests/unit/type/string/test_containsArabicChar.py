"""Unit tests for containsArabicChar."""

import pytest
from py_libraries.type.string import containsArabicChar


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('abc', False),
        ('', False),
        ('123 .,;!', False),
        # ARABIC LETTER ALEF
        ('\u0627', True),
        # ARABIC LETTER HAMZA
        ('\u0621', True),
        # ARABIC-INDIC DIGIT ZERO
        ('\u0660', True),
        # ARABIC LETTER KEHEH
        ('\u06a9', True),
        # ARABIC LETTER BEH WITH THREE DOTS HORIZONTALLY BELOW
        ('\u0750', True),
        # ARABIC LETTER ALEF WASLA ISOLATED FORM
        ('\ufb50', True),
        # ARABIC FATHATAN ISOLATED FORM
        ('\ufe70', True),
        # ARABIC COMMA: no script (Common)
        ('\u060c', False),
        # HEBREW LETTER ALEF: Hebrew
        ('\u05d0', False),
    ],
)
def test_containsArabicChar_known_values(s: str, expected: bool) -> None:
    assert containsArabicChar(s) is expected


@pytest.mark.parametrize(
    'char',
    [
        '\u0627',
        '\u0621',
        '\u0660',
        '\u06a9',
        '\u0750',
        '\ufb50',
        '\ufe70',
    ],
)
def test_containsArabicChar_char_inside_text_without_letters(char: str) -> None:
    assert containsArabicChar(f'123 .,; {char} !? 456') is True
