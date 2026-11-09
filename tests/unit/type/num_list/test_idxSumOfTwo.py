"""Unit tests for idxSumOfTwo."""

import itertools
from collections.abc import Sequence

import numpy as np
import pytest

from py_libraries.type.num_list import idxSumOfTwo


def hasPair(nums: Sequence[int], target: int) -> bool:
    """Reference implementation: test every pair of positions.

    Args:
        nums: Integers to scan.
        target: Sum to reach.

    Returns:
        True if two different positions hold values that sum to `target`.
    """
    return any(a + b == target for a, b in itertools.combinations(nums, 2))


@pytest.mark.parametrize(
    ('nums', 'target', 'expected'),
    [
        ([2, 7, 11, 15], 9, [0, 1]),
        ([3, 2, 4], 6, [1, 2]),
        ([3, 3], 6, [0, 1]),
        ([-3, 4, 3, 90], 0, [0, 2]),
        ([0, 4, 3, 0], 0, [0, 3]),
        ([1, 2, 3], 7, []),
        ([5], 10, []),
        ([], 0, []),
    ],
)
def test_known_values(nums: list[int], target: int, expected: list[int]) -> None:
    """Return the indices of two elements that sum to the target.

    Args:
        nums: Integers to scan.
        target: Sum to reach.
        expected: Expected indices, empty if there is none.
    """
    assert idxSumOfTwo(nums, target) == expected


def test_an_element_cannot_be_used_twice() -> None:
    """Do not pair an element with itself."""
    assert idxSumOfTwo([3, 4], 6) == []


@pytest.mark.parametrize('seed', range(5))
def test_matches_brute_force(seed: int) -> None:
    """Find a valid pair exactly when one exists.

    Args:
        seed: Seed of the random generator that fills the list.
    """
    rng = np.random.default_rng(seed)

    for _ in range(100):
        nums = rng.integers(-6, 7, size=rng.integers(0, 9)).tolist()
        target = int(rng.integers(-10, 11))

        result = idxSumOfTwo(nums, target)

        if hasPair(nums, target):
            i, j = result
            assert 0 <= i < j < len(nums)
            assert nums[i] + nums[j] == target
        else:
            assert result == []
