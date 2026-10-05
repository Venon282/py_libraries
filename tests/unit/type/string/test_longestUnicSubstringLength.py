import random

import pytest
from py_libraries.type.string import longestUnicSubstringLength


def bruteForce(s: str) -> int:
    """Reference implementation: test every substring."""
    best = 0
    for start in range(len(s)):
        for end in range(start, len(s)):
            if len(set(s[start : end + 1])) == end - start + 1:
                best = max(best, end - start + 1)
    return best


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('abcabcbb', 3),
        ('bbbbb', 1),
        ('pwwkew', 3),
        ('abba', 2),
        ('abcdef', 6),
        ('a', 1),
        ('', 0),
    ],
)
def test_longestUnicSubstringLength_known_values(s: str, expected: int) -> None:
    assert longestUnicSubstringLength(s) == expected


def test_longestUnicSubstringLength_matches_brute_force() -> None:
    rng = random.Random(0)

    for _ in range(300):
        s = ''.join(rng.choice('abcd') for _ in range(rng.randint(0, 12)))

        assert longestUnicSubstringLength(s) == bruteForce(s)
