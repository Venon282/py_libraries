"""Unit tests for containsEthiopicChar."""

import pytest
from py_libraries.type.string import containsEthiopicChar


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('abc', False),
        ('', False),
        ('123 .,;!', False),
        # ETHIOPIC SYLLABLE HA
        ('\u1200', True),
        # ETHIOPIC SYLLABLE SEBATBEIT MWA
        ('\u1380', True),
        # ETHIOPIC SYLLABLE LOA
        ('\u2d80', True),
        # ETHIOPIC SYLLABLE TTHU
        ('\uab01', True),
        # HANGUL JONGSEONG SSANGNIEUN: Hangul
        ('\u11ff', False),
        # CANADIAN SYLLABICS HYPHEN: Canadian Aboriginal
        ('\u1400', False),
    ],
)
def test_containsEthiopicChar_known_values(s: str, expected: bool) -> None:
    assert containsEthiopicChar(s) is expected


@pytest.mark.parametrize(
    'char',
    [
        '\u1200',
        '\u1380',
        '\u2d80',
        '\uab01',
    ],
)
def test_containsEthiopicChar_char_inside_text_without_letters(char: str) -> None:
    assert containsEthiopicChar(f'123 .,; {char} !? 456') is True
