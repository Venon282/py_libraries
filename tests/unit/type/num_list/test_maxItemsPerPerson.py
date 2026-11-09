"""Unit tests for maxItemsPerPerson."""

from collections.abc import Sequence

import numpy as np
import pytest

from py_libraries.type.num_list import maxItemsPerPerson


def linearScan(piles: Sequence[int], k: int) -> int:
    """Reference implementation: try each amount from the largest pile down.

    Args:
        piles: Non-negative item counts.
        k: Number of persons, positive.

    Returns:
        The largest amount that serves `k` persons, or 0.
    """
    for amount in range(max(piles, default=0), 0, -1):
        if sum(pile // amount for pile in piles) >= k:
            return amount
    return 0


@pytest.mark.parametrize(
    ('piles', 'k', 'expected'),
    [
        ([5, 8, 6], 3, 5),
        ([2, 5], 11, 0),
        ([4, 7, 5], 16, 1),
        ([4, 7, 5], 17, 0),
        ([1, 1, 1], 3, 1),
        ([10], 1, 10),
        ([10], 3, 3),
        ([10], 10, 1),
        ([10], 11, 0),
        ([0, 6], 2, 3),
        ([], 1, 0),
    ],
)
def test_known_values(piles: list[int], k: int, expected: int) -> None:
    """Find the largest equal share that serves every person.

    Args:
        piles: Item count of each pile.
        k: Number of persons.
        expected: Expected number of items per person.
    """
    assert maxItemsPerPerson(piles, k) == expected


@pytest.mark.parametrize('seed', range(5))
def test_matches_linear_scan(seed: int) -> None:
    """Agree with the exhaustive scan on random piles.

    Args:
        seed: Seed of the random generator that builds the piles.
    """
    rng = np.random.default_rng(seed)

    for _ in range(100):
        piles = rng.integers(0, 30, size=rng.integers(0, 8)).tolist()
        k = int(rng.integers(1, 40))

        assert maxItemsPerPerson(piles, k) == linearScan(piles, k)


def test_more_persons_never_increase_the_share() -> None:
    """Give a smaller or equal share when persons are added."""
    piles = [9, 14, 6, 20]

    shares = [maxItemsPerPerson(piles, k) for k in range(1, 30)]

    assert shares == sorted(shares, reverse=True)


def test_share_serves_k_persons_and_one_more_item_does_not() -> None:
    """Return a share that is feasible and cannot be increased."""
    piles = [7, 11, 13, 5]
    k = 6

    share = maxItemsPerPerson(piles, k)

    assert sum(pile // share for pile in piles) >= k
    assert sum(pile // (share + 1) for pile in piles) < k
