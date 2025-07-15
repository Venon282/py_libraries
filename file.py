from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

def computeGlobalMeanVariance(data_dir, pattern="*.txt", delimiter=None, verbose=False):
    """
    Welford's algorithm
    Compute the global mean and standard deviation of all numeric values
    in files matching the given pattern under data_dir, using Welford's algorithm
    and NumPy for flexible file reading.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing data files.
    pattern : str, optional
        Glob pattern to match files (default '*.txt').
    delimiter : str or None, optional
        Delimiter for np.loadtxt (e.g. ',', '\t'); None means any whitespace.

    Returns
    -------
    mean : float
        The computed global mean.
    variance : float
        The computed global variance (population). sqrt(variance) gives the std.
    count : int
        Total number of data points processed.
    """
    data_path = Path(data_dir)
    n = 0                     # total count of samples
    mean = 0.0                # running mean
    M2 = 0.0                  # sum of squares of differences from the current mean

    iterator = data_path.glob(pattern) if not verbose else tqdm(data_path.glob(pattern), desc="Processing files", unit="file", total=len(list(data_path.glob(pattern))), mininterval=1)
    for filepath in iterator:
        # load entire file into a NumPy array, respecting the given delimiter
        # assumes files contain only numeric columns
        try:
            arr = np.loadtxt(filepath, delimiter=delimiter, dtype=np.float64)
        except ValueError as e:
            raise ValueError(f"Could not parse {filepath}: {e}")

        # flatten in case of multi-column files
        flat = arr.ravel()

        # update Welford accumulators for each element
        for x in flat:
            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            M2 += delta * delta2

    if n == 0:
        raise ValueError("No data points found in any file.")

    variance = M2 / n

    return mean, variance, n

def excelSheetsToCsv(
    excel_path: Path | str,
    output_dir: Path | str,
    sep: str = ",",
    encoding: str = "utf-8",
    include_index: bool = False,
    sheet_name_to_filename: dict[str, str] | None = None,
    **to_csv_kwargs
) -> None:
    """
    Read an Excel file and export each sheet to its own CSV file.

    :param excel_path:      Path to the .xlsx file.
    :param output_dir:      Directory where CSVs will be written (created if needed).
    :param sep:             Field delimiter for the CSV (default: ",").
    :param encoding:        File encoding for the CSV (default: "utf-8").
    :param include_index:   Whether to write row indices into the CSV (default: False).
    :param sheet_name_to_filename:
                             Optional mapping from sheet names -> desired filenames (without extension).
                             If a sheet is not in the dict, its sheet name is used (sanitized).
    :param to_csv_kwargs:   Any additional keyword args passed straight to DataFrame.to_csv().
                            e.g. header=True/False, date_format="...", quoting=csv.QUOTE_ALL, etc.

    :raises FileNotFoundError: If the given Excel file doesn’t exist.
    """
    # Prepare paths
    excel_path = Path(excel_path)
    if not excel_path.is_file():
        raise FileNotFoundError(f"No such Excel file: {excel_path!s}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load workbook (pandas will infer engine)
    workbook = pd.ExcelFile(excel_path)

    for sheet in workbook.sheet_names:
        # Read sheet
        df = pd.read_excel(workbook, sheet_name=sheet)

        # Determine filename: use mapping if provided, else sheet name
        if sheet_name_to_filename and sheet in sheet_name_to_filename:
            name = sheet_name_to_filename[sheet]
        else:
            # sanitize: keep alnum, space, dot, underscore, dash
            name = "".join(c if c.isalnum() or c in (" ", ".", "_", "-") else "_" for c in sheet).strip()
            if not name:
                name = f"sheet_{workbook.sheet_names.index(sheet)}"

        csv_path = output_dir / f"{name}.csv"
        
        # Export
        df.to_csv(
            csv_path,
            sep=sep,
            encoding=encoding,
            index=include_index,
            **to_csv_kwargs
        )