"""Unit tests for toExcel."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from py_libraries.file.h5 import toExcel


def test_writes_every_dataset_to_its_own_sheet(tmp_path: Path) -> None:
    """Write each dataset to a sheet named after its path."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset('top', data=np.array([10, 20]))
        f.create_dataset('raw/a', data=np.array([[1.0, 2.0]]))

    excel = tmp_path / 'out.xlsx'
    with h5py.File(path, 'r') as f:
        toExcel(f, excel_path=excel)

    sheets = pd.read_excel(excel, sheet_name=None)
    assert set(sheets) == {'top', 'raw_a'}
    np.testing.assert_array_equal(sheets['top'].to_numpy().ravel(), [10, 20])
    np.testing.assert_array_equal(sheets['raw_a'].to_numpy(), [[1.0, 2.0]])


def test_sheet_name_is_truncated_to_the_limit(tmp_path: Path) -> None:
    """Truncate a long path to the maximum sheet name length."""
    path = tmp_path / 't.h5'
    long_name = 'a' * 50
    with h5py.File(path, 'w') as f:
        f.create_dataset(long_name, data=np.array([1, 2]))

    excel = tmp_path / 'out.xlsx'
    with h5py.File(path, 'r') as f:
        toExcel(f, excel_path=excel, max_title_chars=31)

    sheets = pd.read_excel(excel, sheet_name=None)
    assert len(sheets) == 1
    (sheet_name,) = sheets
    assert len(sheet_name) <= 31


def test_a_large_dataset_is_split_into_several_sheets(tmp_path: Path) -> None:
    """Split a dataset that exceeds the row limit into several sheets."""
    path = tmp_path / 't.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=np.arange(10))

    excel = tmp_path / 'out.xlsx'
    with h5py.File(path, 'r') as f:
        toExcel(f, excel_path=excel, max_rows=4)

    sheets = pd.read_excel(excel, sheet_name=None)
    total = sum(len(df) for df in sheets.values())
    assert total == 10
    assert len(sheets) > 1
