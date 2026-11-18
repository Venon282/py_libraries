"""Unit tests for getName."""

import h5py
from py_libraries.file.h5 import getName


def test_returns_the_name_of_a_top_level_dataset(h5file: h5py.File) -> None:
    """Return the last element of the internal name."""
    assert getName(h5file['top']) == 'top'


def test_returns_the_name_of_a_nested_dataset(h5file: h5py.File) -> None:
    """Drop the parent path of a nested dataset."""
    assert getName(h5file['raw/a']) == 'a'


def test_returns_the_name_of_a_group(h5file: h5py.File) -> None:
    """Return the name of a group, not its parent path."""
    assert getName(h5file['raw/deep']) == 'deep'


def test_returns_an_empty_string_for_the_root(h5file: h5py.File) -> None:
    """Return an empty string for the root group."""
    assert getName(h5file) == ''
