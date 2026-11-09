import pytest
from py_libraries.other.HappyString import HappyString


@pytest.mark.parametrize(
    ('string', 'expected'),
    [
        ('', True),
        ('a', True),
        ('ab', True),
        ('abc', True),
        ('aba', True),
        ('abab', True),
        ('abcba', True),
        ('aa', False),
        ('aab', False),
        ('abba', False),
    ],
)
def test_isHappyString_known_values(string: str, expected: bool) -> None:
    assert HappyString().isHappyString(string) is expected


def test_isHappyString_rejects_chars_outside_the_allowed_set() -> None:
    happy = HappyString('ab')

    assert happy.isHappyString('ba') is True
    assert happy.isHappyString('abc') is False
    assert happy.isHappyString('') is True
