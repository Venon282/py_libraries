"""Unit tests for extend."""

from pathlib import Path

import h5py
import numpy as np
from py_libraries.file.h5 import extend


def test_creates_a_new_dataset(tmp_path: Path) -> None:
    """Create the dataset with the shape of the values when it is missing."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f:
        extend(f, 'data', [[1, 2], [3, 4]])

    with h5py.File(path, 'r') as f:
        assert f['data'].shape == (2, 2)
        np.testing.assert_array_equal(f['data'][:], [[1, 2], [3, 4]])
        assert f['data'].maxshape[0] is None


def test_appends_to_an_existing_dataset(tmp_path: Path) -> None:
    """Grow the dataset along the first axis when it already exists."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f:
        extend(f, 'data', [1, 2])
        extend(f, 'data', [3, 4])

    with h5py.File(path, 'r') as f:
        np.testing.assert_array_equal(f['data'][:], [1, 2, 3, 4])


def test_scalar_is_stored_as_a_single_row(tmp_path: Path) -> None:
    """Wrap a scalar into a 1D array of one element."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f:
        extend(f, 'data', 7)

    with h5py.File(path, 'r') as f:
        assert f['data'].shape == (1,)
        np.testing.assert_array_equal(f['data'][:], [7])


def test_string_values_are_stored_as_variable_length_strings(tmp_path: Path) -> None:
    """Store string values so that they are read back as variable-length strings."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f:
        extend(f, 'names', ['a', 'bb'])
        extend(f, 'names', ['ccc'])

    with h5py.File(path, 'r') as f:
        assert h5py.check_dtype(vlen=f['names'].dtype) is str
        assert list(f['names'][:]) == [b'a', b'bb', b'ccc']


def test_explicit_dtype_is_used(tmp_path: Path) -> None:
    """Use the given dtype instead of inferring it from the values."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f:
        extend(f, 'data', [1, 2], dtype=np.float32)

    with h5py.File(path, 'r') as f:
        assert f['data'].dtype == np.float32
        np.testing.assert_array_equal(f['data'][:], [1.0, 2.0])


def test_append_keeps_the_dtype_of_the_existing_dataset(tmp_path: Path) -> None:
    """Cast the appended values to the dtype of the existing dataset."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f:
        extend(f, 'data', [1, 2], dtype=np.int32)
        extend(f, 'data', [3, 4])

    with h5py.File(path, 'r') as f:
        assert f['data'].dtype == np.int32
        np.testing.assert_array_equal(f['data'][:], [1, 2, 3, 4])
