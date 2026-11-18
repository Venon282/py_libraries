"""Unit tests for hasDataset."""

from pathlib import Path

import h5py
from py_libraries.file.h5 import hasDataset


def test_true_when_a_top_level_dataset_exists(h5file: h5py.File) -> None:
    """Return True when the group holds a dataset directly."""
    assert hasDataset(h5file) is True


def test_false_for_a_group_that_only_holds_subgroups(tmp_path: Path) -> None:
    """Return False when no dataset sits directly under the group."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f:
        inner = f.create_group('g').create_group('inner')
        inner.create_dataset('d', data=[1])

    with h5py.File(path, 'r') as f:
        assert hasDataset(f['g']) is False
        assert hasDataset(f['g'], rec=True) is True


def test_true_when_recursing_into_a_group_with_nested_datasets(
    h5file: h5py.File,
) -> None:
    """Return True when a subgroup holds a dataset."""
    assert hasDataset(h5file['raw'], rec=True) is True


def test_false_for_an_empty_group(h5file: h5py.File) -> None:
    """Return False when no dataset is found at all."""
    assert hasDataset(h5file['empty']) is False
    assert hasDataset(h5file['empty'], rec=True) is False
