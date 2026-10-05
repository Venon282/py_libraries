import itertools

import pytest
from py_libraries.type.string import isInterleave


def allInterleavings(s1: str, s2: str) -> set[str]:
    """Build every string made by merging s1 and s2 and keeping their orders."""
    total = len(s1) + len(s2)
    result = set()
    for positions in itertools.combinations(range(total), len(s1)):
        first, second = iter(s1), iter(s2)
        chars = [next(first) if k in positions else next(second) for k in range(total)]
        result.add(''.join(chars))
    return result


@pytest.mark.parametrize(
    ('s1', 's2', 's3', 'expected'),
    [
        ('aabcc', 'dbbca', 'aadbbcbcac', True),
        ('aabcc', 'dbbca', 'aadbbbaccc', False),
        ('', '', '', True),
        ('a', '', 'a', True),
        ('', 'b', 'b', True),
        ('a', 'b', 'ba', True),
        ('a', 'b', 'ab', True),
        ('ab', 'ab', 'abba', False),
    ],
)
def test_isInterleave_known_values(s1: str, s2: str, s3: str, expected: bool) -> None:
    assert isInterleave(s1, s2, s3) is expected


def test_isInterleave_length_mismatch_is_false() -> None:
    assert not isInterleave('ab', 'c', 'abcd')
    assert not isInterleave('ab', 'c', 'ab')


def test_isInterleave_matches_exhaustive_enumeration() -> None:
    words = [
        ''.join(chars)
        for size in range(4)
        for chars in itertools.product('ab', repeat=size)
    ]

    for s1, s2 in itertools.product(words, repeat=2):
        interleavings = allInterleavings(s1, s2)
        for chars in itertools.product('ab', repeat=len(s1) + len(s2)):
            s3 = ''.join(chars)

            assert isInterleave(s1, s2, s3) == (s3 in interleavings)
