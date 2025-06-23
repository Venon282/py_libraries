from pathlib import Path
import numpy as np
from tqdm import tqdm

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
