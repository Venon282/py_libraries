"""Unit tests for containsMyanmarChar."""

import pytest
from py_libraries.type.string import containsMyanmarChar


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('abc', False),
        ('', False),
        ('123 .,;!', False),
        # MYANMAR LETTER KA
        ('\u1000', True),
        # MYANMAR LETTER A
        ('\u1021', True),
        # MYANMAR SYMBOL SHAN EXCLAMATION
        ('\u109f', True),
        # MYANMAR LETTER SHAN GHA
        ('\ua9e0', True),
        # MYANMAR LETTER KHAMTI GA
        ('\uaa60', True),
        # TIBETAN LETTER KA: Tibetan
        ('\u0f40', False),
        # GEORGIAN CAPITAL LETTER AN: Georgian
        ('\u10a0', False),
    ],
)
def test_containsMyanmarChar_known_values(s: str, expected: bool) -> None:
    assert containsMyanmarChar(s) is expected


@pytest.mark.parametrize(
    'char',
    [
        '\u1000',
        '\u1021',
        '\u109f',
        '\ua9e0',
        '\uaa60',
    ],
)
def test_containsMyanmarChar_char_inside_text_without_letters(char: str) -> None:
    assert containsMyanmarChar(f'123 .,; {char} !? 456') is True
