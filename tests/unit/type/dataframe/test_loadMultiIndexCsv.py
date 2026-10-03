"""Unit tests for loadMultiIndexCsv."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from py_libraries.type.dataframe import loadMultiIndexCsv, saveMultiIndexCsv


def makeIndex(n_levels: int, prefix: str, named: bool) -> pd.Index:
    """Build a 4-label index with one or two levels, optionally named.

    Args:
        n_levels: Number of levels of the index (1 or 2).
        prefix: Prefix of the level names, followed by the level position.
        named: Whether the levels get a name.

    Returns:
        A flat index for one level, a MultiIndex otherwise.
    """
    names = [f'{prefix}{i}' for i in range(n_levels)] if named else None
    if n_levels == 1:
        return pd.Index(['a', 'b', 'c', 'd'], name=names[0] if names else None)
    return pd.MultiIndex.from_product([['a', 'b'], ['x', 'y']], names=names)


def makeFrame(row_levels: int, col_levels: int, named: bool = False) -> pd.DataFrame:
    """Build a 4 x 4 integer frame with the requested index depths.

    Args:
        row_levels: Number of levels of the row index (1 or 2).
        col_levels: Number of levels of the column index (1 or 2).
        named: Whether the index levels get a name.

    Returns:
        The frame, filled with seeded random integers.
    """
    values = np.random.default_rng(0).integers(0, 100, size=(4, 4))
    return pd.DataFrame(
        values,
        index=makeIndex(row_levels, 'r', named),
        columns=makeIndex(col_levels, 'c', named),
    )


@pytest.mark.parametrize(('row_levels', 'col_levels'), [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_round_trip_restores_the_frame(
    tmp_path: Path, row_levels: int, col_levels: int
) -> None:
    """Restore the saved frame with the same values, labels and level counts.

    Args:
        tmp_path: Pytest temporary directory.
        row_levels: Number of levels of the row index.
        col_levels: Number of levels of the column index.
    """
    frame = makeFrame(row_levels, col_levels)
    csv_path = tmp_path / 'frame.csv'
    saveMultiIndexCsv(frame, csv_path)

    loaded = loadMultiIndexCsv(csv_path)

    pd.testing.assert_frame_equal(loaded, frame)
    assert loaded.index.nlevels == row_levels
    assert loaded.columns.nlevels == col_levels


@pytest.mark.parametrize(('row_levels', 'col_levels'), [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_round_trip_of_a_frame_without_rows_keeps_its_structure(
    tmp_path: Path, row_levels: int, col_levels: int
) -> None:
    """Restore the shape, column labels and level counts of a frame without rows.

    Args:
        tmp_path: Pytest temporary directory.
        row_levels: Number of levels of the row index.
        col_levels: Number of levels of the column index.
    """
    frame = makeFrame(row_levels, col_levels).iloc[0:0]
    csv_path = tmp_path / 'frame.csv'
    saveMultiIndexCsv(frame, csv_path)

    loaded = loadMultiIndexCsv(csv_path)

    assert loaded.shape == frame.shape
    assert list(loaded.columns) == list(frame.columns)
    assert loaded.index.nlevels == row_levels
    assert loaded.columns.nlevels == col_levels


def test_round_trip_keeps_level_names_of_multi_index_axes(tmp_path: Path) -> None:
    """Restore the level names of MultiIndex rows and columns.

    Args:
        tmp_path: Pytest temporary directory.
    """
    csv_path = tmp_path / 'frame.csv'
    saveMultiIndexCsv(makeFrame(2, 2, named=True), csv_path)

    loaded = loadMultiIndexCsv(csv_path)

    assert list(loaded.index.names) == ['r0', 'r1']
    assert list(loaded.columns.names) == ['c0', 'c1']


def test_level_counts_come_from_the_metadata_file(tmp_path: Path) -> None:
    """Read a hand-written CSV with the level counts of its metadata file.

    Args:
        tmp_path: Pytest temporary directory.
    """
    csv_path = tmp_path / 'hand_made.csv'
    csv_path.write_text('k1,k2,val\na,x,1\na,y,2\nb,x,3\n')
    (tmp_path / 'hand_made_meta.json').write_text(
        '{"n_row_levels": 2, "n_col_levels": 1}'
    )

    loaded = loadMultiIndexCsv(csv_path)

    expected = pd.DataFrame(
        {'val': [1, 2, 3]},
        index=pd.MultiIndex.from_tuples(
            [('a', 'x'), ('a', 'y'), ('b', 'x')], names=['k1', 'k2']
        ),
    )
    pd.testing.assert_frame_equal(loaded, expected)


def test_accepts_a_str_path(tmp_path: Path) -> None:
    """Accept the CSV path as a str.

    Args:
        tmp_path: Pytest temporary directory.
    """
    frame = makeFrame(2, 2)
    csv_path = tmp_path / 'frame.csv'
    saveMultiIndexCsv(frame, csv_path)

    loaded = loadMultiIndexCsv(str(csv_path))

    pd.testing.assert_frame_equal(loaded, frame)


def test_kwargs_are_forwarded_to_read_csv(tmp_path: Path) -> None:
    """Pass the extra keyword arguments to `pd.read_csv`.

    Args:
        tmp_path: Pytest temporary directory.
    """
    frame = makeFrame(2, 2)
    csv_path = tmp_path / 'frame.csv'
    saveMultiIndexCsv(frame, csv_path)

    loaded = loadMultiIndexCsv(csv_path, nrows=2)

    pd.testing.assert_frame_equal(loaded, frame.iloc[:2])


def test_missing_csv_and_metadata_raises_file_not_found(tmp_path: Path) -> None:
    """Raise FileNotFoundError when neither the CSV nor its metadata exist.

    Args:
        tmp_path: Pytest temporary directory.
    """
    with pytest.raises(FileNotFoundError):
        loadMultiIndexCsv(tmp_path / 'absent.csv')


def test_missing_metadata_raises_file_not_found(tmp_path: Path) -> None:
    """Raise FileNotFoundError when only the CSV file exists.

    Args:
        tmp_path: Pytest temporary directory.
    """
    csv_path = tmp_path / 'frame.csv'
    csv_path.write_text('a,b\n1,2\n')

    with pytest.raises(FileNotFoundError):
        loadMultiIndexCsv(csv_path)


def test_missing_csv_raises_file_not_found(tmp_path: Path) -> None:
    """Raise FileNotFoundError when only the metadata file exists.

    Args:
        tmp_path: Pytest temporary directory.
    """
    (tmp_path / 'frame_meta.json').write_text('{"n_row_levels": 1, "n_col_levels": 1}')

    with pytest.raises(FileNotFoundError):
        loadMultiIndexCsv(tmp_path / 'frame.csv')
