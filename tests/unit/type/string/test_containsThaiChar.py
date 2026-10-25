"""Unit tests for containsThaiChar."""

import pytest
from py_libraries.type.string import containsThaiChar


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('abc', False),
        ('', False),
        ('123 .,;!', False),
        # THAI CHARACTER KO KAI
        ('\u0e01', True),
        # THAI CHARACTER HO NOKHUK
        ('\u0e2e', True),
        # THAI CHARACTER PHINTHU
        ('\u0e3a', True),
        # THAI CHARACTER SARA E
        ('\u0e40', True),
        # THAI CHARACTER KHOMUT
        ('\u0e5b', True),
        # THAI CURRENCY SYMBOL BAHT: baht sign (Common)
        ('\u0e3f', False),
        # LAO LETTER KO: Lao
        ('\u0e81', False),
        # SINHALA LETTER AYANNA: Sinhala
        ('\u0d85', False),
    ],
)
def test_containsThaiChar_known_values(s: str, expected: bool) -> None:
    assert containsThaiChar(s) is expected


@pytest.mark.parametrize(
    'char',
    [
        '\u0e01',
        '\u0e2e',
        '\u0e3a',
        '\u0e40',
        '\u0e5b',
    ],
)
def test_containsThaiChar_char_inside_text_without_letters(char: str) -> None:
    assert containsThaiChar(f'123 .,; {char} !? 456') is True
