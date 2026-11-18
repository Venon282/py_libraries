"""Unit tests for datasetPaths."""

import h5py
from py_libraries.file.h5 import datasetPaths


def test_returns_only_the_top_level_datasets(h5file: h5py.File) -> None:
    """List the datasets that sit directly under the given group."""
    assert datasetPaths(h5file) == ['/top']


def test_recursion_reaches_the_nested_datasets(h5file: h5py.File) -> None:
    """Collect every dataset of the tree when recursing."""
    expected = {'/top', '/raw/a', '/raw/s', '/raw/deep/b'}

    assert set(datasetPaths(h5file, rec=True)) == expected


def test_recursion_of_a_subgroup_starts_from_that_group(h5file: h5py.File) -> None:
    """Collect the datasets under a subgroup when recursing from it."""
    expected = {'/raw/a', '/raw/s', '/raw/deep/b'}

    assert set(datasetPaths(h5file['raw'], rec=True)) == expected


def test_group_without_dataset_returns_an_empty_list(h5file: h5py.File) -> None:
    """Return an empty list when no dataset is found."""
    assert datasetPaths(h5file['empty']) == []
    assert datasetPaths(h5file['empty'], rec=True) == []


def test_paths_are_the_internal_names_of_the_datasets(h5file: h5py.File) -> None:
    """Return the same paths as the `name` attribute of each dataset."""
    paths = datasetPaths(h5file, rec=True)

    assert all(path.startswith('/') for path in paths)
    assert set(paths) == {'/top', '/raw/a', '/raw/s', '/raw/deep/b'}
