"""Unit tests for displayDetails."""

import h5py
from py_libraries.file.h5 import displayDetails


def test_logs_a_group_line_for_each_group(
    h5file: h5py.File, log_records: list[str]
) -> None:
    """Log one `Group:` line per group of the tree."""
    displayDetails(h5file)

    group_lines = [line for line in log_records if line.startswith('Group: ')]
    assert set(group_lines) == {'Group: empty', 'Group: raw', 'Group: raw/deep'}


def test_logs_a_dataset_line_with_shape_and_dtype(
    h5file: h5py.File, log_records: list[str]
) -> None:
    """Log the name, shape and dtype of each dataset."""
    displayDetails(h5file)

    dataset_lines = [line for line in log_records if line.startswith('Dataset: ')]
    assert 'Dataset: top, shape=(3,), dtype=int64' in dataset_lines
    assert 'Dataset: raw/a, shape=(2, 2), dtype=float64' in dataset_lines


def test_logs_nothing_for_an_empty_group(
    h5file: h5py.File, log_records: list[str]
) -> None:
    """Log nothing when the group holds no object."""
    displayDetails(h5file['empty'])

    assert log_records == []
