import numpy as np
import pytest
from py_libraries.type.lst import findIndicesOfN


def test_findIndicesOfN_known_values() -> None:
    values = np.array([[1, 2, 1], [2, 1, 1]])

    np.testing.assert_array_equal(findIndicesOfN(values, 1), [[0, 2], [1, 2]])


def test_findIndicesOfN_indices_point_to_n() -> None:
    values = np.array([[3, 0, 3, 1], [0, 3, 3, 1], [3, 3, 0, 0]])

    indices = findIndicesOfN(values, 3)

    for row, row_indices in zip(values, indices, strict=True):
        assert np.all(row[row_indices] == 3)


def test_findIndicesOfN_absent_value_gives_empty_second_axis() -> None:
    values = np.array([[1, 2], [3, 4]])

    assert findIndicesOfN(values, 9).shape == (2, 0)


def test_findIndicesOfN_rows_with_different_counts_raise_value_error() -> None:
    values = np.array([[1, 2, 1], [2, 1, 3]])

    with pytest.raises(ValueError):
        findIndicesOfN(values, 1)
