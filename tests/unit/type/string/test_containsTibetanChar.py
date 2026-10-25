"""Unit tests for containsTibetanChar."""

import pytest
from py_libraries.type.string import containsTibetanChar


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('abc', False),
        ('', False),
        ('123 .,;!', False),
        # TIBETAN LETTER KA
        ('\u0f40', True),
        # TIBETAN SYLLABLE OM
        ('\u0f00', True),
        # TIBETAN SUBJOINED LETTER KSSA
        ('\u0fb9', True),
        # MYANMAR LETTER KA: Myanmar
        ('\u1000', False),
        # THAI CHARACTER KO KAI: Thai
        ('\u0e01', False),
    ],
)
def test_containsTibetanChar_known_values(s: str, expected: bool) -> None:
    assert containsTibetanChar(s) is expected


@pytest.mark.parametrize(
    'char',
    [
        '\u0f40',
        '\u0f00',
        '\u0fb9',
    ],
)
def test_containsTibetanChar_char_inside_text_without_letters(char: str) -> None:
    assert containsTibetanChar(f'123 .,; {char} !? 456') is True
