"""Unit tests for saveMultiIndexCsv."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from py_libraries.type.dataframe import saveMultiIndexCsv


def makeIndex(n_levels: int) -> pd.Index:
    """Build a 4-label index with one or two levels.

    Args:
        n_levels: Number of levels of the index (1 or 2).

    Returns:
        A flat index for one level, a MultiIndex otherwise.
    """
    if n_levels == 1:
        return pd.Index(['a', 'b', 'c', 'd'])
    return pd.MultiIndex.from_product([['a', 'b'], ['x', 'y']])


def makeFrame(row_levels: int, col_levels: int) -> pd.DataFrame:
    """Build a 4 x 4 integer frame with the requested index depths.

    Args:
        row_levels: Number of levels of the row index (1 or 2).
        col_levels: Number of levels of the column index (1 or 2).

    Returns:
        The frame, filled with the integers 0 to 15.
    """
    values = np.arange(16).reshape(4, 4)
    return pd.DataFrame(
        values, index=makeIndex(row_levels), columns=makeIndex(col_levels)
    )


@pytest.mark.parametrize(('row_levels', 'col_levels'), [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_writes_csv_and_level_counts_metadata(
    tmp_path: Path, row_levels: int, col_levels: int
) -> None:
    """Write the CSV file and a metadata file holding the level counts.

    Args:
        tmp_path: Pytest temporary directory.
        row_levels: Number of levels of the row index.
        col_levels: Number of levels of the column index.
    """
    csv_path = tmp_path / 'frame.csv'

    saveMultiIndexCsv(makeFrame(row_levels, col_levels), csv_path)

    assert csv_path.is_file()
    meta = json.loads((tmp_path / 'frame_meta.json').read_text())
    assert meta == {'n_row_levels': row_levels, 'n_col_levels': col_levels}


@pytest.mark.parametrize(('row_levels', 'col_levels'), [(1, 1), (2, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize('empty', ['rows', 'columns', 'both'])
def test_empty_frame_keeps_level_counts(
    tmp_path: Path, row_levels: int, col_levels: int, empty: str
) -> None:
    """Record the level counts even when the frame has no rows or no columns.

    Args:
        tmp_path: Pytest temporary directory.
        row_levels: Number of levels of the row index.
        col_levels: Number of levels of the column index.
        empty: Which axis is emptied: `rows`, `columns` or `both`.
    """
    rows = slice(0, 0) if empty in ('rows', 'both') else slice(None)
    cols = slice(0, 0) if empty in ('columns', 'both') else slice(None)
    frame = makeFrame(row_levels, col_levels).iloc[rows, cols]

    saveMultiIndexCsv(frame, tmp_path / 'frame.csv')

    meta = json.loads((tmp_path / 'frame_meta.json').read_text())
    assert meta == {'n_row_levels': row_levels, 'n_col_levels': col_levels}


def test_frame_without_any_label_has_single_level_axes(tmp_path: Path) -> None:
    """Count one level per axis for a completely empty frame.

    Args:
        tmp_path: Pytest temporary directory.
    """
    saveMultiIndexCsv(pd.DataFrame(), tmp_path / 'frame.csv')

    meta = json.loads((tmp_path / 'frame_meta.json').read_text())
    assert meta == {'n_row_levels': 1, 'n_col_levels': 1}


@pytest.mark.parametrize(
    ('file_name', 'meta_name'),
    [
        ('data.csv', 'data_meta.json'),
        ('data.v2.csv', 'data.v2_meta.json'),
        ('noext', 'noext_meta.json'),
    ],
)
def test_metadata_file_is_named_after_the_csv_stem(
    tmp_path: Path, file_name: str, meta_name: str
) -> None:
    """Name the metadata file after the CSV stem plus `_meta.json`.

    Args:
        tmp_path: Pytest temporary directory.
        file_name: Name of the CSV file.
        meta_name: Expected name of the metadata file.
    """
    saveMultiIndexCsv(makeFrame(1, 1), tmp_path / file_name)

    assert (tmp_path / meta_name).is_file()


def test_accepts_a_str_path(tmp_path: Path) -> None:
    """Accept the CSV path as a str.

    Args:
        tmp_path: Pytest temporary directory.
    """
    saveMultiIndexCsv(makeFrame(2, 2), str(tmp_path / 'frame.csv'))

    assert (tmp_path / 'frame.csv').is_file()
    assert (tmp_path / 'frame_meta.json').is_file()


def test_kwargs_are_forwarded_to_to_csv(tmp_path: Path) -> None:
    """Pass the extra keyword arguments to `DataFrame.to_csv`.

    Args:
        tmp_path: Pytest temporary directory.
    """
    csv_path = tmp_path / 'frame.csv'

    saveMultiIndexCsv(makeFrame(1, 1), csv_path, sep=';')

    assert csv_path.read_text().splitlines()[0] == ';a;b;c;d'


def test_saving_again_overwrites_the_metadata(tmp_path: Path) -> None:
    """Replace the metadata when the same CSV path is saved again.

    Args:
        tmp_path: Pytest temporary directory.
    """
    csv_path = tmp_path / 'frame.csv'
    saveMultiIndexCsv(makeFrame(2, 2), csv_path)

    saveMultiIndexCsv(makeFrame(1, 1), csv_path)

    meta = json.loads((tmp_path / 'frame_meta.json').read_text())
    assert meta == {'n_row_levels': 1, 'n_col_levels': 1}
