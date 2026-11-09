import random

import pytest
from py_libraries.other.HappyString import HappyString


def countSubstringsBruteforce(string: str, chars: str) -> int:
    total = 0
    for start in range(len(string)):
        for end in range(start + 1, len(string) + 1):
            if all(c in string[start:end] for c in chars):
                total += 1
    return total


@pytest.mark.parametrize(
    ('string', 'expected'),
    [
        ('', 0),
        ('a', 0),
        ('ab', 0),
        ('abc', 1),
        ('bca', 1),
        ('abcabc', 10),
    ],
)
def test_substringsContainingAllChars_known_values(string: str, expected: int) -> None:
    assert HappyString('abc').substringsContainingAllChars(string) == expected


def test_substringsContainingAllChars_matches_bruteforce() -> None:
    rng = random.Random(0)

    for _ in range(50):
        string = ''.join(rng.choice('abc') for _ in range(rng.randint(0, 10)))
        expected = countSubstringsBruteforce(string, 'abc')
        assert HappyString('abc').substringsContainingAllChars(string) == expected


def test_substringsContainingAllChars_ignores_chars_outside_the_allowed_set() -> None:
    # 'c' is not in the allowed set 'ab', so it is ignored: only substrings that
    # contain both 'a' and 'b' are counted ('ab' and 'abc').
    assert HappyString('ab').substringsContainingAllChars('abc') == 2
    assert HappyString('ab').substringsContainingAllChars(
        'abc'
    ) == countSubstringsBruteforce('abc', 'ab')
