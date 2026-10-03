import numpy as np
import pytest
from py_libraries.type.lst import proportions


def test_proportions_known_values() -> None:
    assert proportions([1, 1, 2, 3]) == {'1': 0.5, '2': 0.25, '3': 0.25}


def test_proportions_strings() -> None:
    result = proportions(['a', 'b', 'a'])

    assert result == {'a': pytest.approx(2 / 3), 'b': pytest.approx(1 / 3)}


def test_proportions_sum_to_one_for_numpy_input() -> None:
    values = np.random.default_rng(0).integers(0, 5, size=100)

    assert sum(proportions(values).values()) == pytest.approx(1.0)


def test_proportions_empty_input_returns_empty_dict() -> None:
    assert proportions([]) == {}


def test_proportions_unhashable_elements_raise_type_error() -> None:
    rows = np.array([[1, 2], [3, 4]])

    with pytest.raises(TypeError):
        proportions(rows)
