import numpy as np
from py_libraries.type.lst import findIndexOfN


def test_findIndexOfN_known_values() -> None:
    values = np.array([[1, 2, 3], [3, 3, 3], [4, 5, 6]])

    np.testing.assert_array_equal(findIndexOfN(values, 3), [2, 0, -1])


def test_findIndexOfN_matches_list_index_on_random_rows() -> None:
    values = np.random.default_rng(0).integers(0, 4, size=(20, 6))

    result = findIndexOfN(values, 2)

    for row, index in zip(values.tolist(), result, strict=True):
        assert index == (row.index(2) if 2 in row else -1)


def test_findIndexOfN_no_rows_gives_empty_result() -> None:
    assert findIndexOfN(np.zeros((0, 3)), 1).size == 0
