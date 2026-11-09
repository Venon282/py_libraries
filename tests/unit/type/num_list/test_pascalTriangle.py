"""Unit tests for pascalTriangle."""

import math

import pytest

from py_libraries.type.num_list import pascalTriangle


def test_known_rows() -> None:
    """Build the first five rows of the triangle."""
    assert pascalTriangle(5) == [
        [1],
        [1, 1],
        [1, 2, 1],
        [1, 3, 3, 1],
        [1, 4, 6, 4, 1],
    ]


def test_single_row() -> None:
    """Return only the first row."""
    assert pascalTriangle(1) == [[1]]


@pytest.mark.parametrize('n_rows', [0, -1, -10])
def test_less_than_one_row_still_gives_the_first_row(n_rows: int) -> None:
    """Always build the first row.

    Args:
        n_rows: Zero or negative number of rows.
    """
    assert pascalTriangle(n_rows) == [[1]]


@pytest.mark.parametrize('n_rows', [1, 2, 7, 30])
def test_values_are_the_binomial_coefficients(n_rows: int) -> None:
    """Hold `comb(k, j)` at row k and position j.

    Args:
        n_rows: Number of rows to build.
    """
    triangle = pascalTriangle(n_rows)

    assert len(triangle) == n_rows
    for k, row in enumerate(triangle):
        assert row == [math.comb(k, j) for j in range(k + 1)]


@pytest.mark.parametrize('n_rows', [5, 12])
def test_rows_are_symmetric_and_sum_to_powers_of_two(n_rows: int) -> None:
    """Read the same in both directions, and sum to 2 ** k.

    Args:
        n_rows: Number of rows to build.
    """
    for k, row in enumerate(pascalTriangle(n_rows)):
        assert row == row[::-1]
        assert sum(row) == 2**k
