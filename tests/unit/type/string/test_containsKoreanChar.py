"""Unit tests for containsKoreanChar."""

import pytest
from py_libraries.type.string import containsKoreanChar


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('abc', False),
        ('', False),
        ('123 .,;!', False),
        # HANGUL SYLLABLE GA
        ('\uac00', True),
        # HANGUL SYLLABLE HIH
        ('\ud7a3', True),
        # HANGUL CHOSEONG KIYEOK
        ('\u1100', True),
        # HANGUL LETTER KIYEOK
        ('\u3131', True),
        # HANGUL JUNGSEONG O-YEO
        ('\ud7b0', True),
        # HALFWIDTH HANGUL FILLER
        ('\uffa0', True),
        # CJK UNIFIED IDEOGRAPH-4E00: Han (Hanja)
        ('\u4e00', False),
        # HIRAGANA LETTER A: Hiragana
        ('\u3042', False),
        # ETHIOPIC SYLLABLE HA: Ethiopic
        ('\u1200', False),
    ],
)
def test_containsKoreanChar_known_values(s: str, expected: bool) -> None:
    assert containsKoreanChar(s) is expected


@pytest.mark.parametrize(
    'char',
    [
        '\uac00',
        '\ud7a3',
        '\u1100',
        '\u3131',
        '\ud7b0',
        '\uffa0',
    ],
)
def test_containsKoreanChar_char_inside_text_without_letters(char: str) -> None:
    assert containsKoreanChar(f'123 .,; {char} !? 456') is True
