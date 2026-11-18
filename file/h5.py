"""Helpers to inspect, extend and export HDF5 files built with h5py."""

import os
import shutil
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from ..other.loggingUtils import getLogger

logger = getLogger(__name__)


def datasetPaths(h5: h5py.File | h5py.Group, rec: bool = False) -> list[str]:
    """Return the internal paths of the datasets of a group or file.

    Args:
        h5: The HDF5 group or file to inspect.
        rec: Whether to descend into subgroups.

    Returns:
        The list of dataset paths.
    """
    return [obj.name for obj in iDatasets(h5, rec)]


def iDatasetPaths(h5: h5py.File | h5py.Group, rec: bool = False) -> Iterator[str]:
    """Yield the internal paths of the datasets of a group or file.

    Args:
        h5: The HDF5 group or file to inspect.
        rec: Whether to descend into subgroups.

    Yields:
        The paths of the datasets, one by one.
    """
    for obj in iDatasets(h5, rec):
        yield obj.name


def datasets(h5: h5py.File | h5py.Group, rec: bool = False) -> list[h5py.Dataset]:
    """Return the datasets of a group or file.

    Args:
        h5: The HDF5 group or file to inspect.
        rec: Whether to descend into subgroups.

    Returns:
        The list of datasets.
    """
    return list(iDatasets(h5, rec))


def iDatasets(h5: h5py.File | h5py.Group, rec: bool = False) -> Iterator[h5py.Dataset]:
    """Yield the datasets of a group or file.

    Args:
        h5: The HDF5 group or file to inspect.
        rec: Whether to descend into subgroups.

    Yields:
        The datasets, one by one.
    """
    if rec:
        q = deque([h5])

        while q:
            group = q.popleft()

            for obj in group.values():
                if isinstance(obj, h5py.Dataset):
                    yield obj
                elif isinstance(obj, h5py.Group):
                    q.append(obj)
    else:
        for obj in h5.values():
            if isinstance(obj, h5py.Dataset):
                yield obj


def hasDataset(h5: h5py.File | h5py.Group, rec: bool = False) -> bool:
    """Check whether a group or file contains at least one dataset.

    Args:
        h5: The HDF5 group or file to inspect.
        rec: Whether to descend into subgroups.

    Returns:
        True if at least one dataset is found, False otherwise.
    """
    try:
        next(iDatasets(h5, rec))
        return True
    except StopIteration:
        return False


def display(h5: h5py.File | h5py.Group) -> None:
    """Log the internal name of every group and dataset of a group or file.

    Args:
        h5: The HDF5 group or file to display.
    """

    def logName(name: str) -> None:
        logger.info(name)

    h5.visit(logName)


def displayDetails(h5: h5py.File | h5py.Group) -> None:
    """Log the name, shape and dtype of every group and dataset.

    Args:
        h5: The HDF5 group or file to display.
    """

    def displayInfo(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Group):
            logger.info('Group: %s', name)
        elif isinstance(obj, h5py.Dataset):
            logger.info('Dataset: %s, shape=%s, dtype=%s', name, obj.shape, obj.dtype)

    h5.visititems(displayInfo)


def describe(h5: h5py.File | h5py.Group) -> pd.DataFrame:
    """Summarize every dataset of a group or file in a DataFrame.

    Each row corresponds to a dataset, indexed by its internal path, and holds
    the statistics computed by `describeValues` over its values.

    Args:
        h5: The HDF5 group or file to describe.

    Returns:
        A DataFrame indexed by dataset path, with one row per dataset.
    """
    from ..type.lst import describeValues

    stats: dict[str, Any] = {}

    def collect(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            stats[name] = describeValues(obj[:])

    h5.visititems(collect)
    return pd.DataFrame.from_dict(stats, orient='index')


def iterate(
    h5: h5py.File | h5py.Group,
    path: str | list[str],
    sep: str = '/',
) -> Iterator[tuple[str, h5py.Group | h5py.Dataset]]:
    """Traverse an HDF5 file or group with a path string that supports wildcards.

    Args:
        h5: An open h5py.File or h5py.Group object.
        path: A path string, or a list of path strings to traverse in turn
            (e.g. `'raw/*'`, `'data/group1/**'`, `'raw/**/data'`,
            `'measurements'`). Supported wildcards:
            - `*` matches every element at the current level.
            - `**` matches every element recursively.
            - `[a,b]` matches the subset of elements listed between the brackets.
            - `~[a,b]` matches every element except the ones listed.
        sep: Path separator (default: `/`).

    Yields:
        Tuples of (internal_path, object) where object is a Group or Dataset.
    """
    if isinstance(path, list):
        for sub_path in path:
            yield from iterate(h5=h5, path=sub_path, sep=sep)
        return

    # Split and clean path
    path_parts = [p for p in path.split(sep) if p]

    def _walk(
        node: h5py.File | h5py.Group,
        parts: list[str],
        root: str = '',
    ) -> Iterator[tuple[str, h5py.Group | h5py.Dataset]]:
        # No more parts: yield the current node
        if not parts:
            yield root.rstrip(sep), node
            return

        part = parts[0]

        # Deep recursive elements case
        if part == '**':
            # Browse recursively all the content
            for key, obj in node.items():
                if isinstance(obj, h5py.Group):
                    # Keep going down with ** and stay at this parts level
                    yield from _walk(obj, parts, root + key + sep)
                    # Then continue the walk (drop **) if there are more parts
                    yield from _walk(obj, parts[1:], root + key + sep)
                else:
                    # A dataset reached by a bare ** is wanted
                    if len(parts) == 1:
                        yield root + key, obj
            return

        # All elements case
        if part == '*':
            for key in node:
                obj = node[key]
                yield from _walk(obj, parts[1:], root + key + sep)
            return

        # A list of elements
        if part.startswith('[') and part.endswith(']'):
            for key in part[1:-1].split(','):
                key = key.strip('\'" ')
                if key in node:
                    yield from _walk(node[key], parts[1:], root + key + sep)
            return

        # A list of excluded elements
        if part.startswith('~[') and part.endswith(']'):
            undesired_keys = [k.strip('\'" ') for k in part[2:-1].split(',')]
            for key in node:
                if key in undesired_keys:
                    continue
                obj = node[key]
                yield from _walk(obj, parts[1:], root + key + sep)
            return

        # Normal case
        if part in node:
            obj = node[part]
            if len(parts) == 1:  # last part reached, yield the node
                yield root + part, obj
            elif isinstance(obj, h5py.Group):
                yield from _walk(obj, parts[1:], root + part + sep)

    yield from _walk(h5, path_parts, root='')


def extend(
    h5: h5py.File | h5py.Group,
    name: str,
    values: Any,
    dtype: Any = None,
) -> None:
    """Create or append data to an HDF5 dataset that can grow in rows.

    If the dataset `name` does not exist in the group `h5`, it is created with
    the shape of `values` (first dimension resizable) and an optional `dtype`.
    If it already exists, `values` is appended along the first axis.

    Args:
        h5: The HDF5 group or file where the dataset resides.
        name: Name of the dataset to create or append to.
        values: Data to store. Scalars are converted to 1D arrays.
        dtype: h5py-compatible dtype. If None and the values are string-like, a
            variable-length UTF-8 string type is used automatically.
    """
    # Ensure values are at least 1D
    values = np.array([values]) if np.isscalar(values) else np.array(values)

    values_shape = values.shape

    if dtype is None and values.dtype.kind in {'U', 'S', 'O'}:
        dtype = h5py.string_dtype(encoding='utf-8')
        values = values.astype(str).tolist()

    if name not in h5:
        max_shape = (None,) + values_shape[1:]  # allow unlimited rows

        h5.create_dataset(
            name,
            data=values,
            maxshape=max_shape,
            chunks=True,
            dtype=dtype,
        )
    else:
        dataset = h5[name]
        old_size = dataset.shape[0]
        new_size = old_size + values_shape[0]
        dataset.resize(new_size, axis=0)
        if h5py.check_dtype(vlen=dataset.dtype) is str and isinstance(
            values, np.ndarray
        ):
            values = values.astype(str).tolist()
        dataset[old_size:new_size] = values


def defineSize(
    h5: h5py.File | h5py.Group,
    name: str,
    new_size: int | None = None,
    quantity_to_add: int | None = None,
    data_shape: Sequence[int] | None = None,
    is_str: bool = False,
    dtype: Any = None,
) -> int:
    """Resize an existing dataset, or create a resizable one.

    Provide either `new_size` or `quantity_to_add` (which may be negative).
    `data_shape` is required only when creating a dataset that does not yet
    exist.

    Args:
        h5: The HDF5 group or file where the dataset resides.
        name: Name of the dataset to resize or create.
        new_size: Target number of rows.
        quantity_to_add: Number of rows to add to the current size. May be
            negative.
        data_shape: Shape of the remaining axes. Required when the dataset does
            not exist yet.
        is_str: Whether the dataset holds variable-length UTF-8 strings.
        dtype: h5py-compatible dtype. Overwritten by the string type when
            `is_str` is True. When neither is given, `np.float32` is used,
            which is h5py's default for a dataset created without data.

    Returns:
        The number of rows of the dataset after the resize.

    Raises:
        ValueError: If the dataset does not exist and `data_shape` is None, or
            if neither `new_size` nor `quantity_to_add` is given.
    """
    if name not in h5:
        if data_shape is None:
            raise ValueError(
                f'Dataset {name} does not exist so the data_shape parameter is '
                'required.'
            )

        if is_str:
            if dtype is not None:
                logger.warning(
                    'is_str=True and dtype is provided; dtype is overwritten '
                    'with the string type.'
                )
            dtype = h5py.string_dtype(encoding='utf-8')
        elif dtype is None:
            # h5py's default dtype for a dataset created without data
            dtype = np.float32

        initial_shape = (0,) + tuple(data_shape)
        max_shape = (None,) + tuple(data_shape)

        h5.create_dataset(
            name,
            shape=initial_shape,
            maxshape=max_shape,
            chunks=True,
            dtype=dtype,
        )

    dataset = h5[name]
    if quantity_to_add is not None:
        new_size = dataset.shape[0] + quantity_to_add
    elif new_size is None:
        raise ValueError('Provide either new_size or quantity_to_add.')

    dataset.resize(new_size, axis=0)
    return new_size


def toExcel(
    h5: h5py.File | h5py.Group,
    excel_path: str | Path = './h5_to_excel.xlsx',
    verbose: int = 0,
    max_rows: int = 1_048_576,
    max_cols: int = 16_384,
    max_title_chars: int = 31,
) -> None:
    """Export every dataset of a group or file to an Excel workbook.

    Large datasets are split into several sheets to respect the Excel row and
    column limits. Each dataset is written to one or more sheets, with names
    derived from the HDF5 path. 1D datasets are treated as single-column
    sheets, and sheet names are truncated to the maximum allowed length.

    Args:
        h5: The HDF5 file or group containing the datasets to export.
        excel_path: Path of the Excel workbook to create.
        verbose: If nonzero, display a progress bar for the dataset iteration.
        max_rows: Maximum number of rows per sheet (Excel limit).
        max_cols: Maximum number of columns per sheet (Excel limit).
        max_title_chars: Maximum number of characters in a sheet name (Excel
            limit).
    """
    with pd.ExcelWriter(excel_path) as writer:
        iterator = tqdm(
            iterate(h5, '**'),
            mininterval=1,
            desc='Exporting datasets',
            disable=not verbose,
        )

        for path, obj in iterator:
            if not isinstance(obj, h5py.Dataset):
                continue

            sheet_name = path.replace('/', '_')[-max_title_chars:]  # Excel limit
            # Determine shape
            rows = len(obj)
            cols = obj.shape[1] if obj.ndim > 1 else 1

            # Loop over row and column chunks
            for i in range(0, rows, max_rows):
                for j in range(0, cols, max_cols):
                    if verbose:
                        iterator.set_description(f'Exporting {path} [{i}:{j}]')

                    # Slice the dataset
                    if obj.ndim == 1:
                        data_chunk = obj[i : i + max_rows]
                    else:
                        data_chunk = obj[i : i + max_rows, j : j + max_cols]

                    sub_sheet_name = (
                        f'{sheet_name}_{i}_{j}'[-max_title_chars:]
                        if i > 0 or j > 0
                        else sheet_name
                    )
                    pd.DataFrame(data_chunk).to_excel(
                        writer, sheet_name=sub_sheet_name, index=False
                    )


def getName(h5: h5py.File | h5py.Group | h5py.Dataset) -> str:
    """Return the last element of the internal name of a group or dataset.

    Args:
        h5: The HDF5 file, group or dataset.

    Returns:
        The name of the object, without its parent path.
    """
    return str(h5.name).split('/')[-1]


def dfToH5(
    df: pd.DataFrame,
    h5_path: str | Path,
    overwrite: bool = True,
    mode: str = 'x',
) -> None:
    """Save a DataFrame to HDF5, one dataset per column.

    String columns (object dtype) are stored as variable-length UTF-8 strings
    so that h5py reads them back as strings.

    Args:
        df: The DataFrame to save.
        h5_path: Path of the HDF5 file to write.
        overwrite: Whether an existing column is replaced. If False, an existing
            column is left untouched.
        mode: Mode passed to `h5py.File`. The default `'x'` creates the file and
            fails if it already exists.
    """
    with h5py.File(h5_path, mode) as hf:
        for col_name in df.columns:
            # Handle an existing column
            if col_name in hf:
                if not overwrite:
                    continue
                del hf[col_name]

            data = df[col_name].values
            # Handle string data (object dtype in pandas)
            if data.dtype.kind == 'O':
                # Convert to a variable-length UTF-8 string type
                hf.create_dataset(
                    col_name, data=data, dtype=h5py.string_dtype(encoding='utf-8')
                )
            # Handle numeric data
            else:
                hf.create_dataset(col_name, data=data)


def makeH5Copy(
    h5_path: str | Path, suffix: str = '_copy', max_tries: int = 1000
) -> str:
    """Create a unique copy of an HDF5 file next to the original.

    Args:
        h5_path: Path of the file to copy.
        suffix: Suffix inserted before the extension of the copy.
        max_tries: Maximum number of candidate names to try before giving up.

    Returns:
        The path of the created copy.

    Raises:
        FileExistsError: If no unique name is found within `max_tries` attempts.
    """
    folder, filename = os.path.split(h5_path)
    base, ext = os.path.splitext(filename)

    # Try friendly names: base_copy.h5, base_copy1.h5, base_copy2.h5, ...
    for i in range(max_tries):
        new_name = f'{base}{suffix}{ext}' if i == 0 else f'{base}{suffix}{i}{ext}'

        dest = os.path.join(folder, new_name)
        if not os.path.exists(dest):
            shutil.copy2(h5_path, dest)
            return dest
    raise FileExistsError(
        f'Could not create a unique copy of {h5_path} after {max_tries} attempts.'
    )


def getH5RowSet(
    h5_path: str | Path,
    columns: Sequence[str],
    chunk_size: int = 100_000,
) -> set[tuple[Any, ...]]:
    """Return the set of unique rows formed by a group of columns.

    The file is read in chunks so that the whole dataset is not loaded into
    memory at once.

    Args:
        h5_path: Path of the HDF5 file.
        columns: Names of the columns that form each row.
        chunk_size: Number of rows read per iteration.

    Returns:
        The set of unique tuples, one per distinct combination of column values.

    Raises:
        KeyError: If a column is not found in the file.
    """
    unique_rows: set[tuple[Any, ...]] = set()

    with h5py.File(h5_path, 'r') as f:
        # Verify the specified columns are present
        missing = [col for col in columns if col not in f]
        if missing:
            raise KeyError(f'Columns not found in file: {missing}')

        h5_size = len(f[columns[0]])
        for i in range(0, h5_size, chunk_size):
            end = min(i + chunk_size, h5_size)
            col_data = [f[col][i:end] for col in columns]
            unique_rows.update(zip(*col_data, strict=False))

    return unique_rows


def mask(
    h5_path: str | Path,
    filters: (
        dict[str, Callable[[np.ndarray], np.ndarray]]
        | list[tuple[str, Callable[[np.ndarray], np.ndarray]]]
    ),
    mask: npt.ArrayLike | None = None,
    chunk_size: int = 10_000,
    verbose: bool = False,
) -> np.ndarray:
    """Filter the row indices of an HDF5 file with a set of predicates.

    All datasets referenced in `filters` must have the same length. The order
    of the filters matters; add the most selective one first.

    Args:
        h5_path: Path of the HDF5 file.
        filters: Mapping from dataset path to a predicate, or a list of such
            pairs. Each predicate receives a numpy array and must return a
            boolean array of the same length.
        mask: Indices to consider. Accepts a boolean array, an integer array,
            or None to include all rows.
        chunk_size: Number of rows read per iteration.
        verbose: Whether to display a progress bar.

    Returns:
        A sorted integer array of the indices that passed all filters.

    Raises:
        TypeError: If `mask` is neither None, a boolean array nor an integer
            array.
    """
    if not len(filters):
        return np.array([], dtype=np.intp)

    if isinstance(filters, dict):
        filters = list(filters.items())

    with h5py.File(h5_path, 'r') as f:
        dataset_size = f[filters[0][0]].shape[0]

        if mask is None:
            active_indices = np.arange(dataset_size, dtype=np.intp)
        else:
            mask = np.asarray(mask)
            if mask.dtype == bool:
                active_indices = np.where(mask)[0].astype(np.intp)
            elif np.issubdtype(mask.dtype, np.integer):
                active_indices = mask.astype(np.intp)
            else:
                raise TypeError(
                    'mask must be None, a boolean array, or an integer array.'
                )

        for dataset_path, predicate in tqdm(
            filters,
            total=len(filters),
            mininterval=1,
            unit=' filter',
            disable=not verbose,
        ):
            dataset = f[dataset_path]
            kept_chunks: list[np.ndarray] = []

            for i in range(0, len(active_indices), chunk_size):
                chunk_indices = active_indices[i : i + chunk_size]
                bool_result = np.asarray(predicate(dataset[chunk_indices]), dtype=bool)
                kept_chunks.append(chunk_indices[bool_result])

            active_indices = (
                np.concatenate(kept_chunks)
                if kept_chunks
                else np.array([], dtype=np.intp)
            )

    return active_indices
