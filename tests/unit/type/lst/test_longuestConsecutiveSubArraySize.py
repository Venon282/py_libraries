from collections.abc import Sequence

import numpy as np
import pytest
from py_libraries.type.lst import longuestConsecutiveSubArraySize


def bruteForce(values: Sequence[object], max_distinct: int) -> int:
    """Reference implementation: test every contiguous subarray."""
    best = 0
    for start in range(len(values)):
        for end in range(start, len(values)):
            if len(set(values[start : end + 1])) <= max_distinct:
                best = max(best, end - start + 1)
    return best


@pytest.mark.parametrize(
    ('values', 'expected'),
    [
        ([1, 2, 1], 3),
        ([0, 1, 2, 2], 3),
        ([1, 2, 3, 2, 2], 4),
        ([3, 3, 3, 1, 2, 1, 1, 2, 3, 3, 4], 5),
        ([4, 4, 4, 4], 4),
        ([7], 1),
        ([], 0),
    ],
)
def test_longuestConsecutiveSubArraySize_known_examples(
    values: list[int], expected: int
) -> None:
    assert longuestConsecutiveSubArraySize(values) == expected


def test_longuestConsecutiveSubArraySize_matches_brute_force() -> None:
    rng = np.random.default_rng(0)

    for _ in range(200):
        values = rng.integers(0, 4, size=rng.integers(0, 13)).tolist()

        assert longuestConsecutiveSubArraySize(values) == bruteForce(values, 2)


def test_longuestConsecutiveSubArraySize_accepts_non_integer_elements() -> None:
    assert longuestConsecutiveSubArraySize(list('aabbcc')) == 4
