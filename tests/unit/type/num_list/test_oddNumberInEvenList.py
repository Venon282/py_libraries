"""Unit tests for oddNumberInEvenList."""

import numpy as np
import pytest

from py_libraries.type.num_list import oddNumberInEvenList


@pytest.mark.parametrize(
    ('nums', 'expected'),
    [
        ([2, 2, 1], 1),
        ([4, 1, 2, 1, 2], 4),
        ([1], 1),
        ([-1, -1, -7], -7),
        ([0, 5, 5], 0),
        ([5, 5, 5], 5),
        ([1, 1, 2, 2], 0),
        ([], 0),
    ],
)
def test_known_values(nums: list[int], expected: int) -> None:
    """Return the element that appears an odd number of times.

    Args:
        nums: Integers to scan.
        expected: Expected result.
    """
    assert oddNumberInEvenList(nums) == expected


@pytest.mark.parametrize('seed', range(5))
def test_single_element_among_shuffled_pairs(seed: int) -> None:
    """Find the unpaired element wherever it is in the list.

    Args:
        seed: Seed of the random generator that builds and shuffles the list.
    """
    rng = np.random.default_rng(seed)
    pairs = rng.integers(-50, 50, size=int(rng.integers(0, 20))).tolist()
    single = int(rng.integers(-50, 50))
    nums = [*pairs, *pairs, single]
    rng.shuffle(nums)

    assert oddNumberInEvenList(nums) == single


def test_input_is_not_modified() -> None:
    """Leave the list untouched."""
    nums = [3, 1, 3]

    oddNumberInEvenList(nums)

    assert nums == [3, 1, 3]
