"""Unit tests for minQueriesForZeroArray."""

from collections.abc import Sequence

import numpy as np
import pytest

from py_libraries.type.num_list import minQueriesForZeroArray


def bruteForce(nums: Sequence[int], queries: Sequence[Sequence[int]]) -> int:
    """Reference implementation: try each number of queries from 0.

    Args:
        nums: Non-negative integers of the array.
        queries: Queries `[left, right, val]`.

    Returns:
        The smallest number of leading queries whose total decrease reaches every
        element, or -1.
    """
    for used in range(len(queries) + 1):
        decrease = [0] * len(nums)
        for left, right, val in queries[:used]:
            for i in range(left, right + 1):
                decrease[i] += val
        if all(d >= n for d, n in zip(decrease, nums, strict=True)):
            return used
    return -1


@pytest.mark.parametrize(
    ('nums', 'queries', 'expected'),
    [
        ([2, 0, 2], [[0, 2, 1], [0, 2, 1], [1, 1, 3]], 2),
        ([4, 3, 2, 1], [[1, 3, 2], [0, 2, 1]], -1),
        ([0], [], 0),
        ([0, 0, 0], [[0, 2, 5]], 0),
        ([1], [[0, 0, 1]], 1),
        ([5], [[0, 0, 2], [0, 0, 2], [0, 0, 2]], 3),
        ([3, 3], [[0, 0, 3], [1, 1, 3]], 2),
        ([1], [], -1),
    ],
)
def test_known_values(nums: list[int], queries: list[list[int]], expected: int) -> None:
    """Find how many leading queries are needed.

    Args:
        nums: Initial array.
        queries: Queries `[left, right, val]`.
        expected: Expected number of queries, or -1.
    """
    assert minQueriesForZeroArray(nums, queries) == expected


@pytest.mark.parametrize('seed', range(10))
def test_matches_brute_force(seed: int) -> None:
    """Agree with the exhaustive search on random arrays and queries.

    Args:
        seed: Seed of the random generator that builds the case.
    """
    rng = np.random.default_rng(seed)

    for _ in range(100):
        size = int(rng.integers(1, 7))
        nums = rng.integers(0, 6, size=size).tolist()
        queries = []
        for _ in range(int(rng.integers(0, 7))):
            left = int(rng.integers(0, size))
            right = int(rng.integers(left, size))
            queries.append([left, right, int(rng.integers(1, 5))])

        assert minQueriesForZeroArray(nums, queries) == bruteForce(nums, queries)


def test_more_queries_never_increase_the_answer() -> None:
    """Keep the same answer when queries are added at the end."""
    nums = [2, 0, 2]
    queries = [[0, 2, 1], [0, 2, 1], [1, 1, 3]]

    assert minQueriesForZeroArray(nums, [*queries, [0, 2, 9]]) == 2


def test_query_beyond_the_array_raises_index_error() -> None:
    """Raise IndexError when a used query ends after the last element."""
    with pytest.raises(IndexError):
        minQueriesForZeroArray([1], [[0, 1, 1]])
