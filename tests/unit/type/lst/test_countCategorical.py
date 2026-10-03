import numpy as np
from py_libraries.type.lst import countCategorical


def test_countCategorical_counts_ones_in_each_column() -> None:
    one_hot = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])

    np.testing.assert_array_equal(countCategorical(one_hot), [2, 1, 0])


def test_countCategorical_ignores_values_other_than_one() -> None:
    values = np.array([[2, 1], [1, 0]])

    np.testing.assert_array_equal(countCategorical(values), [1, 1])


def test_countCategorical_total_equals_number_of_rows_for_one_hot() -> None:
    labels = np.random.default_rng(0).integers(0, 5, size=50)
    one_hot = np.eye(5)[labels]

    assert countCategorical(one_hot).sum() == 50


def test_countCategorical_no_rows_gives_zero_counts() -> None:
    np.testing.assert_array_equal(countCategorical(np.zeros((0, 3))), [0, 0, 0])
