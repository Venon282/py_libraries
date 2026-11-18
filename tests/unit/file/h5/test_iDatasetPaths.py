"""Unit tests for iDatasetPaths."""

import h5py
from py_libraries.file.h5 import iDatasetPaths


def test_yields_only_the_top_level_dataset_paths(h5file: h5py.File) -> None:
    """Yield the paths of the datasets directly under the group."""
    assert list(iDatasetPaths(h5file)) == ['/top']


def test_recursion_yields_every_dataset_path(h5file: h5py.File) -> None:
    """Yield the path of every dataset of the tree when recursing."""
    expected = {'/top', '/raw/a', '/raw/s', '/raw/deep/b'}

    assert set(iDatasetPaths(h5file, rec=True)) == expected


def test_group_without_dataset_yields_nothing(h5file: h5py.File) -> None:
    """Yield nothing when no dataset is found."""
    assert list(iDatasetPaths(h5file['empty'], rec=True)) == []


def test_is_a_generator(h5file: h5py.File) -> None:
    """Yield lazily instead of building a list."""
    import types

    assert isinstance(iDatasetPaths(h5file), types.GeneratorType)
