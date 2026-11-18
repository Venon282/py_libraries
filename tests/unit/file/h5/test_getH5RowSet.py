"""Unit tests for getH5RowSet."""

from pathlib import Path

import h5py
import numpy as np
import pytest
from py_libraries.file.h5 import getH5RowSet


def test_returns_the_set_of_unique_rows(tmp_path: Path) -> None:
    """Return one tuple per distinct combination of column values."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset('x', data=np.array([1, 1, 2, 2]))
        f.create_dataset('y', data=np.array([5, 6, 5, 6]))

    result = getH5RowSet(path, ['x', 'y'])

    assert result == {(1, 5), (1, 6), (2, 5), (2, 6)}


def test_duplicates_are_collapsed(tmp_path: Path) -> None:
    """Collapse repeated combinations into a single tuple."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset('x', data=np.array([1, 1, 1]))
        f.create_dataset('y', data=np.array([2, 2, 2]))

    result = getH5RowSet(path, ['x', 'y'])

    assert result == {(1, 2)}


def test_a_single_column_gives_single_element_tuples(tmp_path: Path) -> None:
    """Return one-element tuples when a single column is requested."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset('x', data=np.array([1, 2, 2]))

    result = getH5RowSet(path, ['x'])

    assert result == {(1,), (2,)}


def test_chunk_size_does_not_change_the_result(tmp_path: Path) -> None:
    """Read the same rows whether in one chunk or several."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset('x', data=np.arange(10))
        f.create_dataset('y', data=np.arange(10) * 2)

    whole = getH5RowSet(path, ['x', 'y'], chunk_size=100_000)
    by_chunk = getH5RowSet(path, ['x', 'y'], chunk_size=3)

    assert whole == by_chunk
    assert len(whole) == 10


def test_missing_column_raises_key_error(tmp_path: Path) -> None:
    """Raise KeyError and name the missing column."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset('x', data=np.array([1]))

    with pytest.raises(KeyError, match='nope'):
        getH5RowSet(path, ['x', 'nope'])
