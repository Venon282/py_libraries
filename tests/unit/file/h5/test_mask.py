"""Unit tests for mask."""

from pathlib import Path

import h5py
import numpy as np
import pytest
from py_libraries.file.h5 import mask


def _write_range(path: Path, length: int = 10) -> None:
    """Write a single integer dataset `x` of the given length.

    Args:
        path: Path of the file to write.
        length: Number of rows of the dataset.
    """
    with h5py.File(path, 'w') as f:
        f.create_dataset('x', data=np.arange(length))


def test_a_dict_of_filters_keeps_the_matching_indices(tmp_path: Path) -> None:
    """Keep the indices that pass the predicate of each dataset."""
    path = tmp_path / 't.h5'
    _write_range(path)

    result = mask(path, {'x': lambda a: a > 5})

    np.testing.assert_array_equal(result, [6, 7, 8, 9])


def test_a_list_of_filters_applies_them_all(tmp_path: Path) -> None:
    """Keep only the indices that pass every filter of the list."""
    path = tmp_path / 't.h5'
    _write_range(path)

    result = mask(path, [('x', lambda a: a > 5), ('x', lambda a: a < 8)])

    np.testing.assert_array_equal(result, [6, 7])


def test_no_filter_returns_an_empty_array(tmp_path: Path) -> None:
    """Return an empty array when no filter is given."""
    path = tmp_path / 't.h5'
    _write_range(path)

    result = mask(path, {})

    assert result.shape == (0,)
    assert result.dtype == np.intp


def test_a_boolean_mask_restricts_the_rows(tmp_path: Path) -> None:
    """Consider only the rows selected by a boolean mask."""
    path = tmp_path / 't.h5'
    _write_range(path)
    even = np.array([i % 2 == 0 for i in range(10)])

    result = mask(path, {'x': lambda a: a > 5}, mask=even)

    np.testing.assert_array_equal(result, [6, 8])


def test_an_integer_mask_restricts_the_rows(tmp_path: Path) -> None:
    """Consider only the rows listed by an integer mask."""
    path = tmp_path / 't.h5'
    _write_range(path)

    result = mask(path, {'x': lambda a: a > 5}, mask=np.array([1, 3, 7, 9]))

    np.testing.assert_array_equal(result, [7, 9])


def test_a_float_mask_raises_type_error(tmp_path: Path) -> None:
    """Raise TypeError for a mask that is neither boolean nor integer."""
    path = tmp_path / 't.h5'
    _write_range(path)

    with pytest.raises(TypeError, match='boolean array, or an integer array'):
        mask(path, {'x': lambda a: a > 5}, mask=np.array([0.5, 1.0]))


def test_chunk_size_does_not_change_the_result(tmp_path: Path) -> None:
    """Keep the same indices whether in one chunk or several."""
    path = tmp_path / 't.h5'
    _write_range(path)

    whole = mask(path, {'x': lambda a: a % 3 == 0}, chunk_size=10_000)
    by_chunk = mask(path, {'x': lambda a: a % 3 == 0}, chunk_size=2)

    np.testing.assert_array_equal(whole, by_chunk)


def test_a_filter_that_matches_nothing_returns_an_empty_array(tmp_path: Path) -> None:
    """Return an empty array when no index passes the filters."""
    path = tmp_path / 't.h5'
    _write_range(path)

    result = mask(path, {'x': lambda a: a > 100})

    assert result.shape == (0,)
