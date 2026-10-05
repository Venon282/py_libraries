import random

import pytest
from py_libraries.type.string import longestPalindrome


def bruteForce(s: str) -> str:
    """Reference implementation: the leftmost longest palindromic substring."""
    best = s[0]
    for start in range(len(s)):
        for end in range(start, len(s)):
            sub = s[start : end + 1]
            if sub == sub[::-1] and len(sub) > len(best):
                best = sub
    return best


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('babad', 'bab'),
        ('cbbd', 'bb'),
        ('a', 'a'),
        ('ac', 'a'),
        ('aaaa', 'aaaa'),
        ('forgeeksskeegfor', 'geeksskeeg'),
    ],
)
def test_longestPalindrome_known_values(s: str, expected: str) -> None:
    assert longestPalindrome(s) == expected


def test_longestPalindrome_matches_brute_force() -> None:
    rng = random.Random(0)

    for _ in range(300):
        s = ''.join(rng.choice('abc') for _ in range(rng.randint(1, 12)))

        assert longestPalindrome(s) == bruteForce(s)


def test_longestPalindrome_empty_string_raises_index_error() -> None:
    with pytest.raises(IndexError):
        longestPalindrome('')
