"""Unit tests for containsSinhalaChar."""

import pytest
from py_libraries.type.string import containsSinhalaChar


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('abc', False),
        ('', False),
        ('123 .,;!', False),
        # SINHALA LETTER AYANNA
        ('\u0d85', True),
        # SINHALA LETTER VAYANNA
        ('\u0dc0', True),
        # SINHALA PUNCTUATION KUNDDALIYA
        ('\u0df4', True),
        # SINHALA ARCHAIC DIGIT ONE
        ('\U000111e1', True),
        # MALAYALAM LETTER A: Malayalam
        ('\u0d05', False),
        # THAI CHARACTER KO KAI: Thai
        ('\u0e01', False),
    ],
)
def test_containsSinhalaChar_known_values(s: str, expected: bool) -> None:
    assert containsSinhalaChar(s) is expected


@pytest.mark.parametrize(
    'char',
    [
        '\u0d85',
        '\u0dc0',
        '\u0df4',
        '\U000111e1',
    ],
)
def test_containsSinhalaChar_char_inside_text_without_letters(char: str) -> None:
    assert containsSinhalaChar(f'123 .,; {char} !? 456') is True
