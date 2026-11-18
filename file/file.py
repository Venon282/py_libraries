"""File reading and export helpers: statistics over text files and Excel to CSV."""

from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


def computeGlobalMeanVariance(
    data_dir: Path | str,
    pattern: str = '*.txt',
    delimiter: str | None = None,
    verbose: bool = False,
    recursive: bool = False,
) -> tuple[float, float, int]:
    """Compute the global mean and variance of the numeric values of a set of files.

    The files matching `pattern` under `data_dir` are read with `np.loadtxt` and
    their values are combined with Welford's algorithm, so the mean and the
    population variance are computed in a single pass without loading every file
    into memory at once.

    Args:
        data_dir: Directory containing the data files.
        pattern: Glob pattern of the files to read.
        delimiter: Delimiter of the files, or None for any whitespace.
        verbose: Whether to display a progress bar.
        recursive: Whether to descend into subdirectories.

    Returns:
        A tuple of the global mean, the population variance and the total number
        of data points.

    Raises:
        ValueError: If a file cannot be parsed, or if no data point is found.
    """
    files = glob(pattern, root_dir=data_dir, recursive=recursive)
    n = 0  # total number of samples
    mean = 0.0  # running mean
    M2 = 0.0  # sum of squared differences from the current mean

    iterator = (
        tqdm(files, desc='Processing files', unit='file', mininterval=1)
        if verbose
        else files
    )
    for filepath in iterator:
        # The files are assumed to hold only numeric columns
        try:
            arr = np.loadtxt(filepath, delimiter=delimiter, dtype=np.float64)
        except ValueError as e:
            raise ValueError(f'Could not parse {filepath}: {e}') from e

        # Flatten in case of multi-column files
        flat = arr.ravel()
        n_b = flat.size
        if n_b == 0:
            continue

        # Combine the batch with the running accumulators
        mean_b = flat.mean()
        var_b = flat.var(ddof=0)
        delta = mean_b - mean
        new_n = n + n_b
        mean += delta * n_b / new_n
        M2 += var_b * n_b + delta**2 * n * n_b / new_n
        n = new_n

    if n == 0:
        raise ValueError('No data points found in any file.')

    variance = M2 / n

    return mean, variance, n


def excelSheetsToCsv(
    excel_path: Path | str,
    output_dir: Path | str,
    sep: str = ',',
    encoding: str = 'utf-8',
    include_index: bool = False,
    sheet_name_to_filename: dict[str, str] | None = None,
    **to_csv_kwargs: Any,
) -> None:
    """Read an Excel file and export each sheet to its own CSV file.

    Args:
        excel_path: Path to the .xlsx file.
        output_dir: Directory where the CSVs are written, created if needed.
        sep: Field delimiter of the CSV.
        encoding: Encoding of the CSV files.
        include_index: Whether the row index is written into the CSV.
        sheet_name_to_filename: Mapping from sheet names to the desired file
            names, without extension. A sheet not in the mapping uses its
            sanitized sheet name.
        to_csv_kwargs: Additional keyword arguments passed to
            `DataFrame.to_csv`.

    Raises:
        FileNotFoundError: If `excel_path` does not exist.
    """
    excel_path = Path(excel_path)
    if not excel_path.is_file():
        raise FileNotFoundError(f'No such Excel file: {excel_path!s}')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # pandas infers the engine from the file extension
    workbook = pd.ExcelFile(excel_path)

    for sheet in workbook.sheet_names:
        df = pd.read_excel(workbook, sheet_name=sheet)

        if sheet_name_to_filename and sheet in sheet_name_to_filename:
            name = sheet_name_to_filename[sheet]
        else:
            # Keep only alnum, space, dot, underscore and dash
            name = ''.join(
                c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in sheet
            ).strip()
            if not name:
                name = f'sheet_{workbook.sheet_names.index(sheet)}'

        csv_path = output_dir / f'{name}.csv'

        df.to_csv(
            csv_path,
            sep=sep,
            encoding=encoding,
            index=include_index,
            **to_csv_kwargs,
        )