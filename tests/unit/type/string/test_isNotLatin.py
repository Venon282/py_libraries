"""Unit tests for isNotLatin."""

import pytest
from py_libraries.type.string import isNotLatin


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('', False),
        ('Hello', False),
        ('Hello, 123!', False),
        ('123', False),
        (' \t\n', False),
        ('.,;:!?()[]', False),
        # LATIN SMALL LETTER E WITH ACUTE: accented letters are Latin
        ('caf\u00e9', False),
        # LATIN SMALL LETTER E followed by COMBINING ACUTE ACCENT (Inherited)
        ('e\u0301', False),
        # FULLWIDTH LATIN CAPITAL LETTER A
        ('\uff21', False),
        # GRINNING FACE: a symbol, not a letter
        ('\U0001f600', False),
        # MATHEMATICAL BOLD CAPITAL A: a letter of no particular script (Common)
        ('\U0001d400', False),
        # KATAKANA-HIRAGANA PROLONGED SOUND MARK: a letter of no script (Common)
        ('\u30fc', False),
        # ARABIC-INDIC DIGIT ZERO: a digit, not a letter
        ('\u0660', False),
        # HIRAGANA LETTER A
        ('\u3042', True),
        # CJK ideographs (Han)
        ('\u4f60\u597d', True),
        # GREEK SMALL LETTER ALPHA after Latin letters
        ('abc\u03b1', True),
        # CYRILLIC letters before and after Latin text
        ('\u043c\u0438\u0440 hello \u043c\u0438\u0440', True),
        # DEVANAGARI LETTER A inside digits and punctuation
        ('1, 2, \u0905, 3.', True),
    ],
)
def test_isNotLatin_known_values(s: str, expected: bool) -> None:
    assert isNotLatin(s) is expected


@pytest.mark.parametrize(
    'char',
    [
        # GREEK SMALL LETTER ALPHA
        '\u03b1',
        # CYRILLIC CAPITAL LETTER A
        '\u0410',
        # ARMENIAN CAPITAL LETTER AYB
        '\u0531',
        # GEORGIAN LETTER AN
        '\u10d0',
        # HEBREW LETTER ALEF
        '\u05d0',
        # ARABIC LETTER ALEF
        '\u0627',
        # DEVANAGARI LETTER A
        '\u0905',
        # THAI CHARACTER KO KAI
        '\u0e01',
        # KHMER LETTER KA
        '\u1780',
        # LAO LETTER KO
        '\u0e81',
        # MYANMAR LETTER KA
        '\u1000',
        # TIBETAN LETTER KA
        '\u0f40',
        # SINHALA LETTER AYANNA
        '\u0d85',
        # ETHIOPIC SYLLABLE HA
        '\u1200',
        # MONGOLIAN LETTER A
        '\u1820',
        # HANGUL SYLLABLE GA
        '\uac00',
        # HIRAGANA LETTER SMALL A
        '\u3041',
        # CJK UNIFIED IDEOGRAPH-4F60
        '\u4f60',
    ],
)
def test_isNotLatin_letter_of_another_script(char: str) -> None:
    assert isNotLatin(char) is True
    assert isNotLatin(f'abc {char} 123') is True


@pytest.mark.parametrize(
    'char',
    [
        # LATIN SMALL LETTER A
        '\u0061',
        # LATIN CAPITAL LETTER Z
        '\u005a',
        # LATIN SMALL LETTER E WITH ACUTE
        '\u00e9',
        # FEMININE ORDINAL INDICATOR
        '\u00aa',
        # LATIN CAPITAL LETTER A WITH MACRON
        '\u0100',
        # LATIN SMALL LETTER Y WITH STROKE
        '\u024f',
        # FULLWIDTH LATIN CAPITAL LETTER A
        '\uff21',
    ],
)
def test_isNotLatin_latin_letter(char: str) -> None:
    assert isNotLatin(char) is False
    assert isNotLatin(f'123 {char} .,;') is False


@pytest.mark.parametrize(
    's',
    ['', '\u3042', 'abc\u03b1', '\u0416', 'caf\u00e9', '123', 'Hello, 123!'],
)
def test_isNotLatin_unchanged_by_latin_text_and_digits(s: str) -> None:
    assert isNotLatin(s + ' abc XYZ 123') is isNotLatin(s)
    assert isNotLatin('abc XYZ 123 ' + s) is isNotLatin(s)
