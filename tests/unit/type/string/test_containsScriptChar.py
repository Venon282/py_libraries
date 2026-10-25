"""Unit tests for containsScriptChar."""

import pytest
import regex
from py_libraries.type.string import containsChineseChar, containsScriptChar


@pytest.mark.parametrize(
    ('s', 'scripts', 'expected'),
    [
        ('abc', ('Latin',), True),
        ('abc', ('Greek',), False),
        ('', ('Latin',), False),
        ('123 .,;!', ('Latin',), False),
        # GREEK SMALL LETTER ALPHA
        ('\u03b1', ('Greek',), True),
        # CYRILLIC CAPITAL LETTER ZHE
        ('abc\u0416', ('Cyrillic',), True),
        # OL CHIKI LETTER LA: script names may contain an underscore
        ('\u1c5a', ('Ol_Chiki',), True),
        # HIRAGANA LETTER A is not Han
        ('\u3042', ('Han',), False),
        # CJK ideograph
        ('\u4f60', ('Han',), True),
    ],
)
def test_containsScriptChar_known_values(
    s: str, scripts: tuple[str, ...], expected: bool
) -> None:
    assert containsScriptChar(s, *scripts) is expected


@pytest.mark.parametrize(
    ('s', 'scripts', 'expected'),
    [
        # CYRILLIC CAPITAL LETTER ZHE matches the second script
        ('abc\u0416', ('Greek', 'Cyrillic'), True),
        # The order of the scripts does not matter
        ('abc\u0416', ('Cyrillic', 'Greek'), True),
        # HEBREW LETTER ALEF matches neither script
        ('\u05d0', ('Greek', 'Cyrillic'), False),
        # A repeated script is harmless
        ('abc', ('Latin', 'Latin'), True),
        # Every script is searched, not only the first one
        ('\u3042', ('Latin', 'Hiragana'), True),
    ],
)
def test_containsScriptChar_several_scripts(
    s: str, scripts: tuple[str, ...], expected: bool
) -> None:
    assert containsScriptChar(s, *scripts) is expected


@pytest.mark.parametrize('script', ['Latin', 'latin', 'LATIN', 'lAtIn'])
def test_containsScriptChar_script_name_ignores_case(script: str) -> None:
    assert containsScriptChar('abc', script) is True
    assert containsScriptChar('\u03b1', script) is False


@pytest.mark.parametrize(
    's',
    [
        '',
        'abc',
        '\u4e00',
        '\u9fff',
        # YIJING HEXAGRAM FOR THE CREATIVE HEAVEN: just below the CJK range
        '\u4dff',
        # YI SYLLABLE IT: just above the CJK range
        '\ua000',
        'ab\u4f60',
    ],
)
def test_containsScriptChar_han_matches_containsChineseChar_on_basic_range(
    s: str,
) -> None:
    assert containsScriptChar(s, 'Han') is containsChineseChar(s)


def test_containsScriptChar_without_script_raises() -> None:
    with pytest.raises(ValueError, match='At least one script name is required'):
        containsScriptChar('abc')


@pytest.mark.parametrize(
    'script',
    [
        'Klingonish',
        '',
        # A name must not be able to alter the structure of the pattern.
        'Latin}\\p{Greek',
        'Latin]',
    ],
)
def test_containsScriptChar_unknown_script_raises(script: str) -> None:
    with pytest.raises(ValueError, match='Unknown Unicode script'):
        containsScriptChar('abc', script)


def test_containsScriptChar_unknown_script_raises_even_after_a_match() -> None:
    with pytest.raises(ValueError, match="Unknown Unicode script: 'Klingonish'"):
        containsScriptChar('abc', 'Latin', 'Klingonish')


def test_containsScriptChar_unknown_script_keeps_original_cause() -> None:
    with pytest.raises(ValueError, match='Unknown Unicode script') as excinfo:
        containsScriptChar('abc', 'Klingonish')

    assert isinstance(excinfo.value.__cause__, regex.error)
