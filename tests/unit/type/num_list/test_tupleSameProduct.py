"""Unit tests for tupleSameProduct."""

import itertools
from collections.abc import Sequence

import numpy as np
import pytest

from py_libraries.type.num_list import tupleSameProduct


def bruteForce(nums: Sequence[int]) -> int:
    """Reference implementation: test every ordered tuple of four positions.

    Args:
        nums: Distinct positive integers.

    Returns:
        The number of ordered tuples (a, b, c, d) with a * b == c * d.
    """
    return sum(
        1
        for a, b, c, d in itertools.permutations(range(len(nums)), 4)
        if nums[a] * nums[b] == nums[c] * nums[d]
    )


@pytest.mark.parametrize(
    ('nums', 'expected'),
    [
        ([2, 3, 4, 6], 8),
        ([1, 2, 4, 5, 10], 16),
        ([1, 2, 3, 4, 5, 6], 16),
        ([2, 3, 5, 7], 0),
        ([1, 2, 3], 0),
        ([5], 0),
        ([], 0),
    ],
)
def test_known_values(nums: list[int], expected: int) -> None:
    """Count the ordered tuples with two pairs of equal product.

    Args:
        nums: Distinct positive integers.
        expected: Expected count.
    """
    assert tupleSameProduct(nums) == expected


@pytest.mark.parametrize('seed', range(8))
def test_matches_brute_force(seed: int) -> None:
    """Agree with the exhaustive search on random distinct integers.

    Args:
        seed: Seed of the random generator that picks the integers.
    """
    rng = np.random.default_rng(seed)
    size = int(rng.integers(0, 9))
    nums = rng.choice(np.arange(1, 25), size=size, replace=False).tolist()

    assert tupleSameProduct(nums) == bruteForce(nums)


def test_order_of_the_elements_does_not_matter() -> None:
    """Give the same count for any permutation of the list."""
    nums = [1, 2, 3, 4, 6, 8, 12]
    expected = tupleSameProduct(nums)

    assert tupleSameProduct(nums[::-1]) == expected
    assert tupleSameProduct(sorted(nums, key=lambda n: n % 5)) == expected


def test_result_is_a_multiple_of_eight() -> None:
    """Count tuples by groups of eight, one for each arrangement of two pairs."""
    assert tupleSameProduct([1, 2, 3, 4, 6, 8, 12, 24]) % 8 == 0
