"""Unit tests for iDatasets."""

import types

import h5py
from py_libraries.file.h5 import iDatasets


def test_yields_only_the_top_level_datasets(h5file: h5py.File) -> None:
    """Yield the datasets directly under the group."""
    result = list(iDatasets(h5file))

    assert [obj.name for obj in result] == ['/top']
    assert all(isinstance(obj, h5py.Dataset) for obj in result)


def test_recursion_yields_every_dataset(h5file: h5py.File) -> None:
    """Yield every dataset of the tree when recursing."""
    expected = {'/top', '/raw/a', '/raw/s', '/raw/deep/b'}

    assert {obj.name for obj in iDatasets(h5file, rec=True)} == expected


def test_group_without_dataset_yields_nothing(h5file: h5py.File) -> None:
    """Yield nothing when no dataset is found."""
    assert list(iDatasets(h5file['empty'], rec=True)) == []


def test_is_a_generator(h5file: h5py.File) -> None:
    """Yield lazily instead of building a list."""
    assert isinstance(iDatasets(h5file), types.GeneratorType)
