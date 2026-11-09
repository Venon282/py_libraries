"""Unit tests for uniqueTriplesSumAt0."""

import itertools
from collections.abc import Sequence

import numpy as np
import pytest

from py_libraries.type.num_list import uniqueTriplesSumAt0


def bruteForce(nums: Sequence[int]) -> list[list[int]]:
    """Reference implementation: test every combination of three positions.

    Args:
        nums: Integers to scan.

    Returns:
        The unique ascending triplets summing to 0, in lexicographic order.
    """
    triples = {
        tuple(sorted(combination))
        for combination in itertools.combinations(nums, 3)
        if sum(combination) == 0
    }
    return [list(triple) for triple in sorted(triples)]


@pytest.mark.parametrize(
    ('nums', 'expected'),
    [
        ([-1, 0, 1, 2, -1, -4], [[-1, -1, 2], [-1, 0, 1]]),
        ([0, 0, 0], [[0, 0, 0]]),
        ([0, 0, 0, 0], [[0, 0, 0]]),
        ([-2, 0, 1, 1, 2], [[-2, 0, 2], [-2, 1, 1]]),
        ([0, 1, 1], []),
        ([1, 2, 3], []),
        ([1, -1], []),
        ([], []),
    ],
)
def test_known_values(nums: list[int], expected: list[list[int]]) -> None:
    """Find the unique triplets that sum to zero.

    Args:
        nums: Integers to scan.
        expected: Expected triplets.
    """
    assert uniqueTriplesSumAt0(nums) == expected


@pytest.mark.parametrize('seed', range(5))
def test_matches_brute_force(seed: int) -> None:
    """Agree with the exhaustive search on random lists.

    Args:
        seed: Seed of the random generator that fills the list.
    """
    rng = np.random.default_rng(seed)

    for _ in range(100):
        nums = rng.integers(-5, 6, size=rng.integers(0, 13)).tolist()

        assert uniqueTriplesSumAt0(nums) == bruteForce(nums)


def test_input_is_not_modified() -> None:
    """Leave the list and its order untouched."""
    nums = [3, -1, 0, -2, 1, 2]

    uniqueTriplesSumAt0(nums)

    assert nums == [3, -1, 0, -2, 1, 2]


def test_order_of_the_input_does_not_matter() -> None:
    """Give the same triplets for any permutation of the list."""
    nums = [-4, -1, -1, 0, 1, 2, 3, -2]

    assert uniqueTriplesSumAt0(nums[::-1]) == uniqueTriplesSumAt0(nums)
