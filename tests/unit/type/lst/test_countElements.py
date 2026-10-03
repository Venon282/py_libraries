import numpy as np
import pytest
from py_libraries.type.lst import countElements


def test_countElements_known_values() -> None:
    assert countElements([1, 1, 2, 3]) == {'1': 2, '2': 1, '3': 1}


def test_countElements_strings() -> None:
    assert countElements(['a', 'b', 'a']) == {'a': 2, 'b': 1}


def test_countElements_total_equals_input_length() -> None:
    values = np.random.default_rng(0).integers(0, 5, size=100)

    assert sum(countElements(values).values()) == 100


def test_countElements_empty_input_returns_empty_dict() -> None:
    assert countElements([]) == {}


def test_countElements_unhashable_elements_raise_type_error() -> None:
    rows = np.array([[1, 2], [3, 4]])

    with pytest.raises(TypeError):
        countElements(rows)
