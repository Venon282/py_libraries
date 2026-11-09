"""Unit tests for mergeIntervals."""

import copy

import numpy as np
import pytest

from py_libraries.type.num_list import mergeIntervals


def covers(intervals: list[list[int]], point: float) -> bool:
    """Check whether a point belongs to the union of closed intervals.

    Args:
        intervals: Closed intervals `[start, end]`.
        point: Point to test.

    Returns:
        True if the point is inside at least one interval.
    """
    return any(start <= point <= end for start, end in intervals)


@pytest.mark.parametrize(
    ('intervals', 'expected'),
    [
        ([[1, 3], [2, 6], [8, 10], [15, 18]], [[1, 6], [8, 10], [15, 18]]),
        ([[1, 4], [4, 5]], [[1, 5]]),
        ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),
        ([[8, 10], [1, 3], [2, 6]], [[1, 6], [8, 10]]),
        ([[1, 10], [2, 3], [4, 5]], [[1, 10]]),
        ([[1, 4], [0, 4]], [[0, 4]]),
        ([[1, 4], [2, 3]], [[1, 4]]),
        ([[5, 7]], [[5, 7]]),
        ([[2, 2], [2, 2]], [[2, 2]]),
        ([], []),
    ],
)
def test_known_values(intervals: list[list[int]], expected: list[list[int]]) -> None:
    """Merge the overlapping and touching intervals, sorted by start.

    Args:
        intervals: Closed intervals to merge.
        expected: Expected merged intervals.
    """
    assert mergeIntervals(intervals) == expected


def test_input_is_not_modified() -> None:
    """Leave the intervals and their order untouched."""
    intervals = [[8, 10], [1, 3], [2, 6]]
    original = copy.deepcopy(intervals)

    mergeIntervals(intervals)

    assert intervals == original


def test_result_does_not_share_lists_with_the_input() -> None:
    """Return new lists, so that editing the result keeps the input intact."""
    intervals = [[1, 2], [5, 6]]

    result = mergeIntervals(intervals)
    result[0][1] = 99

    assert intervals == [[1, 2], [5, 6]]


def test_tuples_are_accepted() -> None:
    """Accept intervals given as tuples, and return lists."""
    assert mergeIntervals([(1, 3), (2, 4)]) == [[1, 4]]


@pytest.mark.parametrize('seed', range(5))
def test_result_is_sorted_disjoint_and_covers_the_same_points(seed: int) -> None:
    """Keep the union of the intervals, in a minimal sorted and disjoint form.

    Args:
        seed: Seed of the random generator that picks the intervals.
    """
    rng = np.random.default_rng(seed)
    intervals = []
    for _ in range(int(rng.integers(1, 10))):
        start = int(rng.integers(0, 30))
        intervals.append([start, start + int(rng.integers(0, 6))])

    result = mergeIntervals(intervals)

    assert all(start <= end for start, end in result)
    assert all(a[1] < b[0] for a, b in zip(result, result[1:], strict=False))
    for point in np.arange(-1, 40, 0.5):
        assert covers(result, float(point)) == covers(intervals, float(point))
