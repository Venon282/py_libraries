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
    ],
)
def test_containsChineseChar_known_values(s: str, expected: bool) -> None:
    assert containsChineseChar(s) is expected
