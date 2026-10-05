import os.path
import random

import pytest
from py_libraries.type.string import longestCommonPrefix


@pytest.mark.parametrize(
    ('strs', 'expected'),
    [
        (['flower', 'flow', 'flight'], 'fl'),
        (['dog', 'racecar', 'car'], ''),
        (['abc'], 'abc'),
        (['abc', 'abc', 'abc'], 'abc'),
        (['', 'a'], ''),
        (['ab', 'abc'], 'ab'),
    ],
)
def test_longestCommonPrefix_known_values(strs: list[str], expected: str) -> None:
    assert longestCommonPrefix(strs) == expected


def test_longestCommonPrefix_matches_os_path_commonprefix() -> None:
    rng = random.Random(0)

    for _ in range(200):
        strs = [
            ''.join(rng.choice('ab') for _ in range(rng.randint(0, 6)))
            for _ in range(rng.randint(1, 5))
        ]

        assert longestCommonPrefix(strs) == os.path.commonprefix(strs)


def test_longestCommonPrefix_empty_list_raises_index_error() -> None:
    with pytest.raises(IndexError):
        longestCommonPrefix([])
