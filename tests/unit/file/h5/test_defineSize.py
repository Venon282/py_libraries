"""Unit tests for defineSize."""

from pathlib import Path

import h5py
import pytest
from py_libraries.file.h5 import defineSize


def test_creates_an_empty_resizable_dataset(tmp_path: Path) -> None:
    """Create a dataset with zero rows and an unbounded first dimension."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f:
        result = defineSize(f, 'data', new_size=0, data_shape=(3,))

    assert result == 0
    with h5py.File(path, 'r') as f:
        assert f['data'].shape == (0, 3)
        assert f['data'].maxshape[0] is None


def test_grows_an_existing_dataset_with_new_size(tmp_path: Path) -> None:
    """Resize an existing dataset to the requested number of rows."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f:
        defineSize(f, 'data', new_size=2, data_shape=(2,))
        result = defineSize(f, 'data', new_size=5)

    assert result == 5
    with h5py.File(path, 'r') as f:
        assert f['data'].shape == (5, 2)


def test_grows_an_existing_dataset_with_quantity_to_add(tmp_path: Path) -> None:
    """Add rows to the current size when `quantity_to_add` is given."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f:
        defineSize(f, 'data', new_size=2, data_shape=(2,))
        result = defineSize(f, 'data', quantity_to_add=3)

    assert result == 5
    with h5py.File(path, 'r') as f:
        assert f['data'].shape == (5, 2)


def test_shrinks_an_existing_dataset_with_a_negative_quantity(tmp_path: Path) -> None:
    """Shrink the dataset when `quantity_to_add` is negative."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f:
        defineSize(f, 'data', new_size=5, data_shape=(2,))
        result = defineSize(f, 'data', quantity_to_add=-2)

    assert result == 3
    with h5py.File(path, 'r') as f:
        assert f['data'].shape == (3, 2)


def test_quantity_to_add_wins_over_new_size(tmp_path: Path) -> None:
    """Prefer `quantity_to_add` when both are given."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f:
        defineSize(f, 'data', new_size=2, data_shape=(2,))
        result = defineSize(f, 'data', new_size=9, quantity_to_add=1)

    assert result == 3


def test_creates_a_string_dataset_with_is_str(tmp_path: Path) -> None:
    """Create a variable-length string dataset when `is_str` is True."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f:
        defineSize(f, 'names', new_size=0, data_shape=(), is_str=True)

    with h5py.File(path, 'r') as f:
        assert h5py.check_dtype(vlen=f['names'].dtype) is str
        assert f['names'].shape == (0,)


def test_missing_dataset_without_data_shape_raises_value_error(tmp_path: Path) -> None:
    """Raise ValueError when creating a dataset without `data_shape`."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f, pytest.raises(ValueError, match='data_shape'):
        defineSize(f, 'missing', new_size=3)


def test_existing_dataset_without_a_size_raises_value_error(tmp_path: Path) -> None:
    """Raise ValueError when resizing without `new_size` or `quantity_to_add`."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f:
        defineSize(f, 'data', new_size=2, data_shape=(2,))
        with pytest.raises(ValueError, match='new_size or quantity_to_add'):
            defineSize(f, 'data')
