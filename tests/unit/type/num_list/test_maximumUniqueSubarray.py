"""Unit tests for maximumUniqueSubarray."""

from collections.abc import Sequence

import numpy as np
import pytest

from py_libraries.type.num_list import maximumUniqueSubarray


def bruteForce(nums: Sequence[int]) -> int:
    """Reference implementation: test every contiguous subarray.

    Args:
        nums: Positive integers.

    Returns:
        The maximum sum of a subarray without repeated element, 0 if empty.
    """
    best = 0
    for start in range(len(nums)):
        for end in range(start, len(nums)):
            sub = nums[start : end + 1]
            if len(set(sub)) == len(sub):
                best = max(best, sum(sub))
    return best


@pytest.mark.parametrize(
    ('nums', 'expected'),
    [
        ([4, 2, 4, 5, 6], 17),
        ([5, 2, 1, 2, 5, 2, 1, 2, 5], 8),
        ([1, 2, 3, 4], 10),
        ([3, 3, 3], 3),
        ([1], 1),
        ([], 0),
    ],
)
def test_known_values(nums: list[int], expected: int) -> None:
    """Find the best sum of a subarray of distinct elements.

    Args:
        nums: Positive integers to scan.
        expected: Expected sum.
    """
    assert maximumUniqueSubarray(nums) == expected


@pytest.mark.parametrize('seed', range(5))
def test_matches_brute_force(seed: int) -> None:
    """Agree with the exhaustive search on random positive integers.

    Args:
        seed: Seed of the random generator that fills the list.
    """
    rng = np.random.default_rng(seed)

    for _ in range(100):
        nums = rng.integers(1, 8, size=rng.integers(0, 14)).tolist()

        assert maximumUniqueSubarray(nums) == bruteForce(nums)


def test_all_distinct_elements_give_the_total_sum() -> None:
    """Keep the whole list when it has no duplicate."""
    nums = [7, 1, 9, 3, 5]

    assert maximumUniqueSubarray(nums) == sum(nums)
