import pytest
from py_libraries.type.string import containsChineseChar


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('abc', False),
        ('', False),
        ('ab\u4f60', True),
        ('\u4e00', True),
        ('\u9fff', True),
        ('\u4dff', False),
        ('\ua000', False),
        # CJK UNIFIED IDEOGRAPH-3400: first of Extension A, below the old range
        ('\u3400', True),
        # CJK UNIFIED IDEOGRAPH-4DBF: last of Extension A
        ('\u4dbf', True),
        # CJK UNIFIED IDEOGRAPH-20000: first of Extension B (outside the BMP)
        ('\U00020000', True),
        # CJK COMPATIBILITY IDEOGRAPH-F900
        ('\uf900', True),
        # CJK RADICAL REPEAT
        ('\u2e80', True),
        # IDEOGRAPHIC ITERATION MARK: a mark of the Han script
        ('\u3005', True),
        # IDEOGRAPHIC COMMA: CJK punctuation belongs to no script
        ('\u3001', False),
        # HIRAGANA LETTER A
        ('\u3042', False),
        # HANGUL SYLLABLE GA
        ('\uac00', False),
    ],
)
def test_containsChineseChar_known_values(s: str, expected: bool) -> None:
    assert containsChineseChar(s) is expected
