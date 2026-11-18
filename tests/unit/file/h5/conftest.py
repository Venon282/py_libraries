"""Shared fixtures of the h5 tests."""

import logging
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import pytest
from py_libraries.file import h5 as h5_module


@pytest.fixture
def h5file(tmp_path: Path) -> Iterator[h5py.File]:
    """Open a temporary HDF5 file with a fixed nested structure.

    The file holds a 1D dataset at the root, a `raw` group with a 2D dataset and
    a string dataset, and a `raw/deep` group with another 1D dataset.

    Args:
        tmp_path: Pytest temporary directory.

    Yields:
        The open file. It is closed after the test.
    """
    path = tmp_path / 'structure.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset('top', data=np.array([10, 20, 30]))
        raw = f.create_group('raw')
        raw.create_dataset('a', data=np.array([[1.0, 2.0], [3.0, 4.0]]))
        raw.create_dataset('s', data=['x', 'yy'])
        deep = raw.create_group('deep')
        deep.create_dataset('b', data=np.array([1, 2]))
        f.create_group('empty')

    with h5py.File(path, 'r') as f:
        yield f


@pytest.fixture
def h5path(tmp_path: Path) -> Path:
    """Return the path of a temporary HDF5 file with a single dataset.

    The file holds a 1D integer dataset `data` of length 10.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        The path of the created file.
    """
    path = tmp_path / 'flat.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=np.arange(10))
    return path


@pytest.fixture
def log_records() -> Iterator[list[str]]:
    """Collect the messages logged by the h5 module during a test.

    The module logger does not propagate, so a dedicated handler is attached to
    it for the duration of the test.

    Yields:
        The list of formatted log messages, filled as the test runs.
    """
    records: list[str] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _Collector()
    logger = h5_module.logger
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
