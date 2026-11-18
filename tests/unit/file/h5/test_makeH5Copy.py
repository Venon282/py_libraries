"""Unit tests for makeH5Copy."""

from pathlib import Path

import h5py
import numpy as np
import pytest
from py_libraries.file.h5 import makeH5Copy


def _write(path: Path, values: list[int]) -> None:
    """Write a single-dataset HDF5 file.

    Args:
        path: Path of the file to write.
        values: Values of the dataset.
    """
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=np.array(values))


def test_creates_a_copy_next_to_the_original(tmp_path: Path) -> None:
    """Create `base_copy.h5` next to the original file."""
    source = tmp_path / 'base.h5'
    _write(source, [1, 2, 3])

    dest = makeH5Copy(source)

    assert dest == str(tmp_path / 'base_copy.h5')
    with h5py.File(dest, 'r') as f:
        np.testing.assert_array_equal(f['data'][:], [1, 2, 3])


def test_skips_a_name_that_already_exists(tmp_path: Path) -> None:
    """Use `base_copy1.h5` when `base_copy.h5` already exists."""
    source = tmp_path / 'base.h5'
    _write(source, [1])
    _write(tmp_path / 'base_copy.h5', [9])

    dest = makeH5Copy(source)

    assert dest == str(tmp_path / 'base_copy1.h5')


def test_a_custom_suffix_is_used(tmp_path: Path) -> None:
    """Insert the given suffix before the extension of the copy."""
    source = tmp_path / 'base.h5'
    _write(source, [1])

    dest = makeH5Copy(source, suffix='_backup')

    assert dest == str(tmp_path / 'base_backup.h5')


def test_raises_file_exists_error_when_no_name_is_free(tmp_path: Path) -> None:
    """Give up after `max_tries` candidate names are all taken."""
    source = tmp_path / 'base.h5'
    _write(source, [1])
    _write(tmp_path / 'base_copy.h5', [9])
    _write(tmp_path / 'base_copy1.h5', [9])
    _write(tmp_path / 'base_copy2.h5', [9])

    with pytest.raises(FileExistsError):
        makeH5Copy(source, max_tries=3)
