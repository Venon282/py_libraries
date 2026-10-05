import random

import pytest
from py_libraries.type.string import maxOccurenceAdjacentChar


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('abbcccddddeeeeedcba', 5),
        ('leetcode', 2),
        ('triplepillooooow', 5),
        ('hooraaaaaaaaaaay', 11),
        ('aa', 2),
    ],
)
def test_maxOccurenceAdjacentChar_known_values(s: str, expected: int) -> None:
    assert maxOccurenceAdjacentChar(s) == expected


def test_maxOccurenceAdjacentChar_empty_string_gives_zero() -> None:
    assert maxOccurenceAdjacentChar('') == 0


def test_maxOccurenceAdjacentChar_equals_the_longest_built_run() -> None:
    rng = random.Random(0)
    alphabet = 'abcd'

    for _ in range(100):
        lengths = [rng.randint(2, 6) for _ in range(rng.randint(1, 6))]
        s = ''.join(
            alphabet[k % len(alphabet)] * length for k, length in enumerate(lengths)
        )

        assert maxOccurenceAdjacentChar(s) == max(lengths)
