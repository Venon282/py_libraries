"""Unit tests for describe."""

import h5py
import pandas as pd
from py_libraries.file.h5 import describe


def test_returns_a_dataframe_indexed_by_dataset_path(h5file: h5py.File) -> None:
    """Return one row per dataset, indexed by its internal path."""
    result = describe(h5file)

    assert isinstance(result, pd.DataFrame)
    assert set(result.index) == {'top', 'raw/a', 'raw/s', 'raw/deep/b'}


def test_numeric_dataset_row_matches_describeValues(h5file: h5py.File) -> None:
    """Carry the statistics computed by `describeValues` for the dataset."""
    from py_libraries.type.lst import describeValues

    result = describe(h5file)

    assert result.loc['top', 'total_count'] == 3
    assert result.loc['top', 'min'] == 10
    assert result.loc['top', 'max'] == 30
    # The row holds at least the keys of the numeric dataset; the DataFrame
    # also carries the string-only keys (as NaN) because all rows are aligned.
    assert set(describeValues(h5file['top'][:]).keys()) <= set(result.loc['top'].index)


def test_string_dataset_row_is_described_as_strings(h5file: h5py.File) -> None:
    """Describe a string dataset through its string form."""
    result = describe(h5file)

    assert result.loc['raw/s', 'total_count'] == 2
    assert result.loc['raw/s', 'unique_count'] == 2


def test_group_without_dataset_returns_an_empty_dataframe(h5file: h5py.File) -> None:
    """Return an empty DataFrame when no dataset is found."""
    result = describe(h5file['empty'])

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
