"""Utilities to count, smooth, resample and describe lists and arrays of values."""

import math
import statistics
from collections import Counter
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import kurtosis, skew


def proportions(lst: Sequence[Hashable] | np.ndarray) -> dict[str, float]:
    """Compute the proportion of each distinct element of a sequence.

    Args:
        lst: Elements to count.

    Returns:
        Mapping from the string form of each distinct element to its proportion
        in `lst`.

    Raises:
        TypeError: If an element is not hashable.
    """
    return {str(element): list(lst).count(element) / len(lst) for element in set(lst)}


def countElements(lst: Sequence[Hashable] | np.ndarray) -> dict[str, int]:
    """Count the occurrences of each distinct element of a sequence.

    Args:
        lst: Elements to count.

    Returns:
        Mapping from the string form of each distinct element to its number of
        occurrences in `lst`.

    Raises:
        TypeError: If an element is not hashable.
    """
    return {str(element): list(lst).count(element) for element in set(lst)}


def interpolation(
    x_targ: Sequence[float], x: Sequence[float], y: Sequence[float]
) -> list[float]:
    """Linearly interpolate the y values at the target x values.

    Prefer `np.interp`: this implementation can return wrong values when several
    consecutive targets fall in the same interval of `x`. Both `x` and `x_targ`
    are expected in ascending order. Targets outside the range of `x` take the
    nearest edge value of `y`.

    Args:
        x_targ: Target x values to interpolate the y values for.
        x: Original x values.
        y: Original y values corresponding to `x`.

    Returns:
        Interpolated y values, one for each target x value.
    """

    def jup(j: int, x: Sequence[float]) -> int:
        return j if j + 1 >= len(x) else j + 1

    new_y: list[float] = []
    j = 0
    for xt in x_targ:
        if xt < x[j]:
            if j == 0:
                # If our y start later than the need we have,
                # keep the first value of val
                new_y.append(y[j])
            else:
                distance_max = x[j] - x[j - 1]
                distance = x[j] - xt
                ratio = distance / distance_max
                new_y.append(y[j] - (y[j] - y[j - 1]) * ratio)
                j = jup(j, x)
        elif xt > x[j]:
            # catch up the late if j have more than 1 value bellow the need wavelength
            while j + 1 < len(x) and xt > x[j + 1]:
                j = jup(j, x)
            if j + 1 >= len(x):
                # not anymore j value
                # todo improve by checking the curve direction
                new_y.append(y[j])
            else:
                distance_max = x[j + 1] - x[j]
                distance = xt - x[j]
                ratio = distance / distance_max
                new_y.append(y[j] + (y[j + 1] - y[j]) * ratio)
                j = jup(j, x)
        else:  # if the wavelength are equals
            new_y.append(y[j])
            j = jup(j, x)
    return new_y


def smoothMiddle(lst: Sequence[float], window: int = 5) -> list[float]:
    """Smooth values with a moving average centered on each element.

    For an even window, the extra element is taken on the right of the current
    element. The window is truncated at both ends of the sequence.

    Args:
        lst: Values to smooth.
        window: Size of the moving window.

    Returns:
        Smoothed values, with the same length as `lst`.
    """
    shift = max(1, window // 2)
    shiftl, shiftr = (shift, shift) if window % 2 == 1 else (shift - 1, shift)
    return [
        statistics.mean(lst[max(0, i - shiftl) : min(len(lst), i + shiftr + 1)])
        for i, _ in enumerate(lst)
    ]


def countCategorical(lst: np.ndarray) -> np.ndarray:
    """Count the entries equal to 1 in each column of a categorical 2D array.

    Args:
        lst: One-hot encoded 2D array.

    Returns:
        Number of entries equal to 1 in each column.
    """
    counts: np.ndarray = np.sum(lst == 1, axis=0)
    return counts


def findIndicesOfN(lst: np.ndarray, n: int) -> np.ndarray:
    """Find the indices where the element n is present in each row.

    Args:
        lst: 2D array of elements.
        n: Element to find.

    Returns:
        Array with one row per row of `lst`, holding the indices of the
        occurrences of `n` in that row.

    Raises:
        ValueError: If the rows do not all contain `n` the same number of times.
    """
    # Use NumPy to identify locations where elements equal n
    return np.array([np.where(row == n)[0] for row in lst])


def findIndexOfN(lst: np.ndarray, n: int) -> np.ndarray:
    """Find the first index of the element n in each row.

    Args:
        lst: 2D array of elements.
        n: Element to find.

    Returns:
        Array holding, for each row, the index of the first occurrence of `n`, or
        -1 if the row does not contain `n`.
    """
    # Use NumPy to identify the first occurrence of n in each row
    return np.array(
        [np.where(row == n)[0][0] if np.any(row == n) else -1 for row in lst]
    )


def repartition(
    data1: Sequence[object] | np.ndarray,
    data2: Sequence[object] | np.ndarray,
    proportion: float,
) -> tuple[Sequence[object] | np.ndarray, Sequence[object] | np.ndarray]:
    """Divide data between two tables according to a given proportion.

    Args:
        data1: First data table.
        data2: Second data table.
        proportion: Proportion to be used for allocation (e.g. 0.8).

    Returns:
        The part extracted from `data1` and the part extracted from `data2`.
    """
    # Initial data size
    size1 = len(data1)
    size2 = len(data2)

    # Total size of combined data
    total_size = size1 + size2

    # Target size for each table
    target_size1 = math.ceil(proportion * total_size)
    target_size2 = total_size - target_size1  # Complement

    # Calculation of extractions respecting the relative sizes
    if size1 < target_size1:
        # If data1 is too small to meet the proportion, adjust based on size1
        split_size1 = size1
        split_size2 = math.ceil(size1 * (1 - proportion) / proportion)
    elif size2 < target_size2:
        # If data2 is too small to meet the complementary proportion,
        # adjust based on size2
        split_size2 = size2
        split_size1 = math.ceil(size2 * proportion / (1 - proportion))
    else:
        # Both tables have enough data to meet the target proportion
        split_size1 = size1
        split_size2 = size2

    # Split
    split_data1 = data1[:split_size1]
    split_data2 = data2[:split_size2]

    return split_data1, split_data2


def repartitionNbNeed(
    data1: Sequence[object] | np.ndarray,
    data2: Sequence[object] | np.ndarray,
    proportion: float,
) -> tuple[int, int]:
    """Compute how many elements `repartition` extracts from each table.

    Args:
        data1: First data table.
        data2: Second data table.
        proportion: Proportion to be used for allocation (e.g. 0.8).

    Returns:
        The number of elements to extract from `data1` and from `data2`.
    """
    # Initial data size
    size1 = len(data1)
    size2 = len(data2)

    # Total size of combined data
    total_size = size1 + size2

    # Target size for each table
    target_size1 = math.ceil(proportion * total_size)
    target_size2 = total_size - target_size1  # Complement

    # Calculation of extractions respecting the relative sizes
    if size1 < target_size1:
        # If data1 is too small to meet the proportion, adjust based on size1
        return size1, math.ceil(size1 * (1 - proportion) / proportion)
    elif size2 < target_size2:
        # If data2 is too small to meet the complementary proportion,
        # adjust based on size2
        return math.ceil(size2 * proportion / (1 - proportion)), size2
    else:
        # Both tables have enough data to meet the target proportion
        return size1, size2


def describeValues(
    array: npt.ArrayLike,
    chunk_size: int = 1_000_000,
    sample_limit: int = 5_000_000,
    str_most_common_limit: int | None = 5,
) -> dict[str, Any]:
    """Compute descriptive statistics of an array with a bounded memory usage.

    The data is processed in chunks instead of being loaded at once. Numeric
    arrays give numeric statistics; any other dtype is described through the
    string form of its values.

    Args:
        array: Values to describe.
        chunk_size: Number of elements processed per chunk.
        sample_limit: Maximum number of finite values kept to compute the quartiles,
            the median, the number of unique values, the skewness and the kurtosis.
            Above this limit, these statistics become estimates computed on a random
            sample. The other statistics are always exact.
        str_most_common_limit: For non-numeric arrays, number of most common values
            returned. None returns the whole counter.

    Returns:
        For numeric arrays, a mapping with the keys `shape`, `total_count`,
        `count` (finite values), `nan_count`, `pos_inf_count`, `neg_inf_count`,
        `zero_count`, `nan_rate`, `pos_inf_rate`, `neg_inf_rate`,
        `zero_count_rate`, `min`, `25%`, `median`, `mean`, `std`, `var`, `IQR`,
        `75%`, `max`, `range`, `unique_count`, `skewness` and `kurtosis`. The
        statistics computed on finite values are NaN when there is none.
        For other arrays, a mapping with the keys `shape`, `total_count`,
        `unique_count`, `min_length`, `max_length` and `most_common`.

    Raises:
        TypeError: If the array is not numeric and `str_most_common_limit` is
            neither an int nor None.
    """
    arr = np.asarray(array)
    shape = arr.shape
    total_count = arr.size

    # Check if array is numeric
    if np.issubdtype(arr.dtype, np.number):
        nan_count = pos_inf_count = neg_inf_count = zero_count = 0
        finite_count = 0

        # Initialize running stats
        finite_chunks: list[np.ndarray] = []
        finite_min = np.inf
        finite_max = -np.inf
        mean_accum = 0.0
        m2 = 0.0  # for variance (Welford's method)

        # Iterate through array in chunks
        for start in range(0, total_count, chunk_size):
            end = min(start + chunk_size, total_count)
            chunk = arr.flat[start:end]

            # Count infinities / NaNs
            nan_mask = np.isnan(chunk)
            pos_mask = np.isposinf(chunk)
            neg_mask = np.isneginf(chunk)
            finite_mask = np.isfinite(chunk)

            nan_count += int(np.count_nonzero(nan_mask))
            pos_inf_count += int(np.count_nonzero(pos_mask))
            neg_inf_count += int(np.count_nonzero(neg_mask))

            finite_chunk = chunk[finite_mask]
            n = finite_chunk.size
            if n == 0:
                continue

            finite_count += n
            zero_count += int(np.count_nonzero(finite_chunk == 0))

            # Min / max
            finite_min = min(finite_min, np.min(finite_chunk))
            finite_max = max(finite_max, np.max(finite_chunk))

            # Pairwise variance combination.
            # Incremental mean / variance (Chan/Welford)
            prev_count = finite_count - n  # count BEFORE adding this chunk
            delta = finite_chunk.mean() - mean_accum
            mean_accum += delta * n / finite_count
            m2 += (
                finite_chunk.var(ddof=0) * n + delta**2 * prev_count * n / finite_count
            )

            # Store chunk values for quantile/skew/kurtosis (optional)
            finite_chunks.append(finite_chunk)

        # Combine finite values (or sample if too large)
        if finite_chunks:
            if len(finite_chunks) * chunk_size > sample_limit:
                last_rate = len(finite_chunks[-1]) / chunk_size
                size_by_unit = sample_limit / (len(finite_chunks) - 1 + last_rate)
                size_by_unit_int = int(size_by_unit)
                finite_vals = np.concatenate(
                    [
                        np.random.choice(
                            finite_chunks[i],
                            min(size_by_unit_int, len(finite_chunks[i])),
                            replace=False,
                        )
                        for i in range(len(finite_chunks) - 1)
                    ]
                    + [
                        np.random.choice(
                            finite_chunks[-1],
                            int(size_by_unit * last_rate),
                            replace=False,
                        )
                    ]
                )
            else:
                finite_vals = np.concatenate(finite_chunks)
        else:
            finite_vals = np.array([])

        # Rates
        if total_count > 0:
            nan_rate = nan_count / total_count
            pos_inf_rate = pos_inf_count / total_count
            neg_inf_rate = neg_inf_count / total_count
        else:
            nan_rate = pos_inf_rate = neg_inf_rate = np.nan

        # Finite stats
        if finite_count > 0:
            q1 = np.percentile(finite_vals, 25)
            median_val = np.median(finite_vals)
            mean_val = mean_accum
            std_val = np.sqrt(m2 / finite_count)
            var_val = std_val**2
            q3 = np.percentile(finite_vals, 75)
            iqr = q3 - q1
            range_val = finite_max - finite_min
            unique_count = np.unique(finite_vals).size if finite_vals.size else np.nan
            zero_count_rate = zero_count / finite_count
            skewness = skew(finite_vals)
            kurt = kurtosis(finite_vals)
        else:
            finite_min = q1 = median_val = mean_val = std_val = var_val = q3 = np.nan
            finite_max = iqr = range_val = unique_count = zero_count_rate = np.nan
            skewness = kurt = np.nan

        stats = {
            'shape': shape,
            'total_count': int(total_count),
            'count': int(finite_count),
            'nan_count': int(nan_count),
            'pos_inf_count': int(pos_inf_count),
            'neg_inf_count': int(neg_inf_count),
            'zero_count': int(zero_count),
            'nan_rate': nan_rate,
            'pos_inf_rate': pos_inf_rate,
            'neg_inf_rate': neg_inf_rate,
            'zero_count_rate': zero_count_rate,
            'min': finite_min,
            '25%': q1,
            'median': median_val,
            'mean': mean_val,
            'std': std_val,
            'var': var_val,
            'IQR': iqr,
            '75%': q3,
            'max': finite_max,
            'range': range_val,
            'unique_count': unique_count,
            'skewness': skewness,
            'kurtosis': kurt,
        }
    else:
        # String version
        unique_vals: Counter[str] = Counter()
        max_len = 0
        min_len = np.inf

        for start in range(0, total_count, chunk_size):
            end = min(start + chunk_size, total_count)
            chunk = arr.flat[start:end]

            strings = [str(x) for x in chunk]  # ensure all are strings
            lengths = [len(x) for x in strings]
            max_len = max(max_len, max(lengths, default=0))
            min_len = min(min_len, min(lengths, default=0))
            unique_vals.update(strings)

        most_common: Counter[str] | list[tuple[str, int]]
        if str_most_common_limit is None:
            most_common = unique_vals
        elif isinstance(str_most_common_limit, int):
            most_common = unique_vals.most_common(str_most_common_limit)
        else:
            raise TypeError(
                'str_most_common_limit must be either int or None, '
                f'got {type(str_most_common_limit)}.'
            )

        stats = {
            'shape': shape,
            'total_count': int(total_count),
            'unique_count': len(unique_vals),
            'min_length': min_len,
            'max_length': max_len,
            'most_common': most_common,
        }

    return stats


def describe(
    array: npt.ArrayLike,
    min_threshold: float = 1e-4,
    max_threshold: float = 1e5,
    n_decimals: int | None = 4,
) -> None:
    """Print the descriptive statistics of an array, one per line.

    Args:
        array: Values to describe, see `describeValues`.
        min_threshold: Non-zero values with a magnitude below this threshold are
            printed in scientific notation.
        max_threshold: Values with a magnitude greater than or equal to this
            threshold are printed in scientific notation.
        n_decimals: Number of decimals printed. None disables the rounding of
            standard notation values, and scientific notation then uses 2 decimals.
    """
    values = describeValues(array)
    for key, value in values.items():
        # Attempt to calculate the absolute value (for numerical cases)
        try:
            abs_val = abs(value)
        except TypeError:
            abs_val = None

        # Use scientific notation if the value is not zero and its magnitude
        # is less than min_threshold or greater than or equal to max_threshold
        if (
            abs_val is not None
            and value != 0
            and (abs_val < min_threshold or abs_val >= max_threshold)
        ):
            n = 2 if n_decimals is None else n_decimals
            formatted = f'{value:.{n}e}'  # value at scientific format
            if 'e' in formatted:
                parts = formatted.split('e')  # separate the mantissa and exponent
                mantissa = parts[0].rstrip('0').rstrip('.')  # remove useless 0
                exponent = parts[1]
                print(f'{key}: {mantissa}e{exponent}')
            else:
                print(f'{key}: {formatted}')
        else:
            if n_decimals is not None and isinstance(value, (int, float)):
                print(f'{key}: {round(value, n_decimals)}')
            else:
                print(f'{key}: {value}')


def sort(lst: list[str]) -> None:
    """Sort a list of strings in place using a natural order.

    Digit sequences are compared as integers and sort before text, which is
    compared case-insensitively.

    Args:
        lst: Strings to sort. The list is modified in place.
    """
    import re

    def naturalKey(s: str) -> list[tuple[int, int | str]]:
        """
        Transforms string s into a list of tokens:
        - each sequence of digits becomes a tuple (0, integer_value)
        - each sequence of non-numbers becomes a tuple (1, lowercase_string)
        In this way, all numeric blocks (0) come before letter blocks (1).
        """
        tokens = re.findall(r'\d+|\D+', s)
        key: list[tuple[int, int | str]] = []
        for tok in tokens:
            if tok.isdigit():
                key.append((0, int(tok)))
            else:
                key.append((1, tok.lower()))
        return key

    lst.sort(key=naturalKey)


def longuestConsecutiveSubArraySize(lst: Sequence[object], sub_size: int = 2) -> int:
    """Find the size of the longest contiguous subarray with few distinct values.

    With the default `sub_size`, this solves the classic "fruit into baskets"
    problem. The implementation is only reliable for `sub_size=2`.

    Args:
        lst: Elements to scan.
        sub_size: Maximum number of distinct values allowed in the subarray.

    Returns:
        Size of the longest contiguous subarray of `lst` holding at most
        `sub_size` distinct values.
    """
    sub_unic: list[object] = []
    total = 0
    total_cur = 0
    n_prev = 0

    for e in lst:
        # If the element is in the current pair current total +1
        if e in sub_unic:
            total_cur += 1

            # If it's not the last seen element put it as the last
            if e != sub_unic[-1]:
                sub_unic.pop(sub_unic.index(e))
                sub_unic.append(e)
                n_prev = 1
            else:
                n_prev += 1
        # Else if the element is not in the current pair
        else:
            # And the pair array is at it's final size
            if len(sub_unic) == sub_size:
                total = max(total, total_cur)  # Update the total
                sub_unic.pop(0)  # Remove the element from the previous pair
                total_cur = 1 + n_prev
            else:
                total_cur += 1

            n_prev = 1
            sub_unic.append(e)

    return max(total, total_cur)
