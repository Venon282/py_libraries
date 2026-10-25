"""Unit tests for containsJapaneseChar."""

import pytest
from py_libraries.type.string import containsJapaneseChar


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('abc', False),
        ('', False),
        ('123 .,;!', False),
        # HIRAGANA LETTER SMALL A
        ('\u3041', True),
        # HIRAGANA LETTER A
        ('\u3042', True),
        # HIRAGANA DIGRAPH YORI
        ('\u309f', True),
        # KATAKANA LETTER SMALL A
        ('\u30a1', True),
        # KATAKANA DIGRAPH KOTO
        ('\u30ff', True),
        # KATAKANA LETTER SMALL KU
        ('\u31f0', True),
        # HALFWIDTH KATAKANA LETTER KA
        ('\uff76', True),
        # HIRAGANA LETTER ARCHAIC YE
        ('\U0001b001', True),
        # KATAKANA MIDDLE DOT: katakana middle dot (Common)
        ('\u30fb', False),
        # KATAKANA-HIRAGANA PROLONGED SOUND MARK: prolonged sound mark (Common)
        ('\u30fc', False),
        # IDEOGRAPHIC SPACE: ideographic space (Common)
        ('\u3000', False),
        # CJK UNIFIED IDEOGRAPH-4E00: Han (Kanji)
        ('\u4e00', False),
        # HANGUL SYLLABLE GA: Hangul
        ('\uac00', False),
    ],
)
def test_containsJapaneseChar_known_values(s: str, expected: bool) -> None:
    assert containsJapaneseChar(s) is expected


@pytest.mark.parametrize(
    'char',
    [
        '\u3041',
        '\u3042',
        '\u309f',
        '\u30a1',
        '\u30ff',
        '\u31f0',
        '\uff76',
        '\U0001b001',
    ],
)
def test_containsJapaneseChar_char_inside_text_without_letters(char: str) -> None:
    assert containsJapaneseChar(f'123 .,; {char} !? 456') is True
