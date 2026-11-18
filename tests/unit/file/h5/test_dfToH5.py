"""Unit tests for dfToH5."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from py_libraries.file.h5 import dfToH5


def test_writes_one_dataset_per_column(tmp_path: Path) -> None:
    """Store every column of the DataFrame as its own dataset."""
    path = tmp_path / 't.h5'
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [1.5, 2.5, 3.5]})

    dfToH5(df, path)

    with h5py.File(path, 'r') as f:
        assert set(f) == {'a', 'b'}
        np.testing.assert_array_equal(f['a'][:], [1, 2, 3])
        np.testing.assert_array_equal(f['b'][:], [1.5, 2.5, 3.5])


def test_string_column_is_stored_as_variable_length_strings(tmp_path: Path) -> None:
    """Store an object column as variable-length UTF-8 strings."""
    path = tmp_path / 't.h5'
    df = pd.DataFrame({'name': ['a', 'bb', 'ccc']})

    dfToH5(df, path)

    with h5py.File(path, 'r') as f:
        assert h5py.check_dtype(vlen=f['name'].dtype) is str
        assert list(f['name'][:]) == [b'a', b'bb', b'ccc']


def test_default_mode_fails_when_the_file_exists(tmp_path: Path) -> None:
    """Raise OSError when the file already exists and the mode is `x`."""
    path = tmp_path / 't.h5'
    df = pd.DataFrame({'a': [1]})

    dfToH5(df, path)
    with pytest.raises(OSError):
        dfToH5(df, path)


def test_overwrite_replaces_an_existing_column(tmp_path: Path) -> None:
    """Replace a column that already exists when `overwrite` is True."""
    path = tmp_path / 't.h5'
    dfToH5(pd.DataFrame({'a': [1, 2]}), path)

    dfToH5(pd.DataFrame({'a': [9]}), path, mode='r+')

    with h5py.File(path, 'r') as f:
        np.testing.assert_array_equal(f['a'][:], [9])


def test_no_overwrite_keeps_the_existing_column(tmp_path: Path) -> None:
    """Keep a column that already exists when `overwrite` is False."""
    path = tmp_path / 't.h5'
    dfToH5(pd.DataFrame({'a': [1, 2]}), path)

    dfToH5(pd.DataFrame({'a': [9], 'b': [5]}), path, overwrite=False, mode='r+')

    with h5py.File(path, 'r') as f:
        np.testing.assert_array_equal(f['a'][:], [1, 2])
        np.testing.assert_array_equal(f['b'][:], [5])
