"""Unit tests for maxArea."""

from collections.abc import Sequence

import numpy as np
import pytest

from py_libraries.type.num_list import maxArea


def bruteForce(height: Sequence[int]) -> int:
    """Reference implementation: test every pair of lines.

    Args:
        height: Non-negative heights.

    Returns:
        The maximum area, 0 if there are fewer than two lines.
    """
    return max(
        (
            (j - i) * min(height[i], height[j])
            for i in range(len(height))
            for j in range(i + 1, len(height))
        ),
        default=0,
    )


@pytest.mark.parametrize(
    ('height', 'expected'),
    [
        ([1, 8, 6, 2, 5, 4, 8, 3, 7], 49),
        ([1, 1], 1),
        ([4, 3, 2, 1, 4], 16),
        ([1, 2, 1], 2),
        ([0, 0], 0),
        ([5], 0),
        ([], 0),
    ],
)
def test_known_values(height: list[int], expected: int) -> None:
    """Find the container that holds the most water.

    Args:
        height: Heights of the vertical lines.
        expected: Expected area.
    """
    assert maxArea(height) == expected


@pytest.mark.parametrize('seed', range(5))
def test_matches_brute_force(seed: int) -> None:
    """Agree with the exhaustive search on random heights.

    Args:
        seed: Seed of the random generator that fills the list.
    """
    rng = np.random.default_rng(seed)

    for _ in range(100):
        height = rng.integers(0, 21, size=rng.integers(0, 15)).tolist()

        assert maxArea(height) == bruteForce(height)


def test_reversed_list_gives_the_same_area() -> None:
    """Be symmetric: the area does not depend on the direction of the list."""
    height = [3, 9, 4, 1, 8, 2, 6]

    assert maxArea(height[::-1]) == maxArea(height)
