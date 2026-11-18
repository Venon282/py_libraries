"""Unit tests for datasets."""

import h5py
from py_libraries.file.h5 import datasets


def test_returns_only_the_top_level_datasets(h5file: h5py.File) -> None:
    """Return the datasets that sit directly under the given group."""
    result = datasets(h5file)

    assert len(result) == 1
    assert result[0].name == '/top'
    assert isinstance(result[0], h5py.Dataset)


def test_recursion_returns_every_dataset(h5file: h5py.File) -> None:
    """Return every dataset of the tree when recursing."""
    expected = {'/top', '/raw/a', '/raw/s', '/raw/deep/b'}

    assert {obj.name for obj in datasets(h5file, rec=True)} == expected


def test_group_without_dataset_returns_an_empty_list(h5file: h5py.File) -> None:
    """Return an empty list when no dataset is found."""
    assert datasets(h5file['empty'], rec=True) == []


def test_returns_a_list(h5file: h5py.File) -> None:
    """Return a list, not a generator."""
    assert isinstance(datasets(h5file), list)
