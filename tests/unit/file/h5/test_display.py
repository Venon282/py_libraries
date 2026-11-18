"""Unit tests for display."""

import h5py
from py_libraries.file.h5 import display


def test_logs_the_name_of_every_group_and_dataset(
    h5file: h5py.File, log_records: list[str]
) -> None:
    """Log the relative name of every object of the tree."""
    expected = {'empty', 'raw', 'raw/a', 'raw/s', 'raw/deep', 'raw/deep/b', 'top'}

    display(h5file)

    assert set(log_records) == expected


def test_logs_nothing_for_an_empty_group(
    h5file: h5py.File, log_records: list[str]
) -> None:
    """Log nothing when the group holds no object."""
    display(h5file['empty'])

    assert log_records == []


def test_logs_the_subgroup_names_when_called_on_a_subgroup(
    h5file: h5py.File, log_records: list[str]
) -> None:
    """Log the names relative to the given subgroup."""
    expected = {'a', 'deep', 'deep/b', 's'}

    display(h5file['raw'])

    assert set(log_records) == expected
