"""Unit tests for computeMaxAscendingSum."""

from collections.abc import Sequence

import numpy as np
import pytest

from py_libraries.type.number import computeMaxAscendingSum


def bruteForce(nums: Sequence[int]) -> int:
    """Reference implementation: test every contiguous subarray.

    Args:
        nums: Non-empty list of positive integers.

    Returns:
        The maximum sum of a strictly ascending contiguous subarray.
    """
    best = nums[0]
    for start in range(len(nums)):
        for end in range(start, len(nums)):
            sub = nums[start : end + 1]
            if all(a < b for a, b in zip(sub, sub[1:], strict=False)):
                best = max(best, sum(sub))
    return best


@pytest.mark.parametrize(
    ('nums', 'expected'),
    [
        ([10, 20, 30, 5, 10, 50], 65),
        ([10, 20, 30, 40, 50], 150),
        ([12, 17, 15, 13, 10, 11, 12], 33),
        ([100, 10, 1], 100),
        ([5], 5),
        ([1, 1, 1], 1),
        ([3, 2, 1, 4], 5),
    ],
)
def test_known_values(nums: list[int], expected: int) -> None:
    """Return the best sum of a strictly ascending run.

    Args:
        nums: Integers to process.
        expected: Expected result.
    """
    assert computeMaxAscendingSum(nums) == expected


def test_matches_brute_force() -> None:
    """Agree with the exhaustive search on random lists of positive integers."""
    rng = np.random.default_rng(0)

    for _ in range(200):
        nums = rng.integers(1, 10, size=rng.integers(1, 12)).tolist()

        assert computeMaxAscendingSum(nums) == bruteForce(nums)


def test_empty_list_raises_index_error() -> None:
    """Raise IndexError when there is no first element."""
    with pytest.raises(IndexError):
        computeMaxAscendingSum([])
