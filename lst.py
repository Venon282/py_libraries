import pickle
import warnings
import statistics
import joblib
import numpy as np
import math
from collections import Counter
from scipy.stats import skew, kurtosis

def proportions(lst):
    return {str(element): list(lst).count(element) / len(lst) for element in set(lst)}

def countElements(lst):
    return {str(element): list(lst).count(element) for element in set(lst)}

def interpolation(x_targ, x, y):
    """interpolation the y values for the x_targ values
    !!! Linear interpolation, better to use np.interpolate!!!
    Args:
        x_targ (list of float): Target x values to interpret y values for.
        x (list of float): Original x values.
        y (list of float): Original y values corresponding to x values.
    """
    def jup(j, x):
        return j if j+1>=len(x) else j+1

    new_y = []
    j = 0
    for xt in x_targ:
        if xt < x[j]:
            if j == 0:
                # If our y start later than the need we have,
                # keep the first value of val
                new_y.append(y[j])
            else:
                distance_max = x[j] - x[j-1]
                distance = x[j] - xt
                ratio = distance / distance_max
                new_y.append(y[j] - (y[j] - y[j-1]) * ratio)
                j=jup(j, x)
        elif xt > x[j]:
            while j+1 < len(x) and xt > x[j+1]: # catch up the late if j have more than 1 value bellow the need wavelength
                j=jup(j, x)
            if j+1 >= len(x):
                new_y.append(y[j]) # not anymore j value # todo improve by checking the curve direction
            else:
                distance_max = x[j+1] - x[j]
                distance = xt - x[j]
                ratio = distance / distance_max
                new_y.append(y[j] + (y[j+1] - y[j]) * ratio)
                j=jup(j, x)
        else: # if the wavelength are equals
            new_y.append(y[j])
            j=jup(j, x)
    return new_y

def smoothMiddle(lst, window=5):
    """Smooth with the current element place on the middle of the window

    Args:
        lst (_type_): _description_
        window (int, optional): _description_. Defaults to 5. Min is 1.
    """
    new_lst = []
    shift = max(1, window//2)
    shiftl, shiftr = (shift, shift) if window%2==1 else (shift-1, shift)
    return [statistics.mean(lst[max(0, i-shiftl):min(len(lst), i+shiftr+1)]) for i, l in enumerate(lst)]

def countCategorical(lst):
    """Count the number of occurences of each element in the categoricals 2d array

    Args:
        lst (list): List of elements

    Returns:
        dict: {element: count}
    """
    return np.sum(lst == 1, axis=0)

def findIndicesOfN(lst, n):
    """Find the indices of subarrays where the element n is present.

    Args:
        lst (np.ndarray): 2D NumPy array of elements.
        n (int): Element to find.

    Returns:
        list: A list of arrays, where each array contains the indices in the subarray where n is present.
    """
    # Use NumPy to identify locations where elements equal n
    return np.array([np.where(row == n)[0] for row in lst])

def findIndexOfN(lst, n):
    """Find the first index of element n in each sub-array.

    Args:
        lst (np.ndarray): 2D NumPy array of elements.
        n (int): Element to find.

    Returns:
        list: A list of indices, where each index is the first occurrence of n in the respective sub-array.
    """
    # Use NumPy to identify the first occurrence of n in each row
    return np.array([np.where(row == n)[0][0] if np.any(row == n) else -1 for row in lst])


def repartition(data1, data2, proportion):
    """
    Divides data between two tables according to a given proportion.

    Parameters:
    - data1 (list or array): First data table.
    - data2 (list or array): Second data table.
    - proportion (float): Proportion to be used for allocation (e.g. 0.8).

    Returns:
    - split_data1 (list): Part extracted from data1.
    - split_data2 (list): Part extracted from data2.
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
        split_size2 = math.ceil(size1 * (1-proportion) / proportion)
    elif size2 < target_size2:
        # If data2 is too small to meet the complementary proportion, adjust based on size2
        split_size2 = size2
        split_size1 = math.ceil(size2 * proportion / (1-proportion))
    else:
        # Both tables have enough data to meet the target proportion
        split_size1 = size1
        split_size2 = size2

    # Répartition
    split_data1 = data1[:split_size1]
    split_data2 = data2[:split_size2]

    return split_data1, split_data2

def repartitionNbNeed(data1, data2, proportion):
    """
    Divides data between two tables according to a given proportion.

    Parameters:
    - data1 (list or array): First data table.
    - data2 (list or array): Second data table.
    - proportion (float): Proportion to be used for allocation (e.g. 0.8).

    Returns:
    - split_data1 (list): Part extracted from data1.
    - split_data2 (list): Part extracted from data2.
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
        return size1, math.ceil(size1 * (1-proportion) / proportion)
    elif size2 < target_size2:
        # If data2 is too small to meet the complementary proportion, adjust based on size2
        return math.ceil(size2 * proportion / (1-proportion)), size2
    else:
        # Both tables have enough data to meet the target proportion
        return size1, size2

# def describeValues(array):
#     """
#     Similar function to pandas.describe() using numpy.
    
#     Calculate descriptive statistics on a numpy array, including:
#       - shape            : Array dimensions
#       - total_count      : Total number of elements
#       - count            : Number of finite elements (non-NaN and non-±inf)
#       - nan_count        : Number of NaNs
#       - pos_inf_count    : Number of positive infinities
#       - neg_inf_count    : Number of negative infinities
#       - zero_count       : Number of zeros among finite values
#       - nan_rate         : NaN rate (nan_count / total_count)
#       - pos_inf_rate     : Positive infinity rate (pos_inf_count / total_count)
#       - neg_inf_rate     : Negative infinity rate (neg_inf_count / total_count)
#       - zero_count_rate  : Zero count rate (zero_count / total_count, among finite values)
#       - min              : Minimum value (among finite values)
#       - 25% (q1)         : First quartile (25th percentile)
#       - median           : Median (50th percentile)
#       - mean             : Mean
#       - std              : Standard deviation
#       - var              : Variance
#       - IQR              : Interquartile range (q3 - q1)
#       - 75% (q3)         : Third quartile (75th percentile)
#       - max              : Maximum value
#       - range            : Range (max - min)
#       - unique_count     : Number of unique values ​​among finite values
#       - skewness         : Distribution skewness (if Scipy is available)
#       - kurtosis         : Distribution kurtosis (if Scipy is available)
      
#     Parameters:
#     -----------
#     array : array_like
#         Numpy array (can be multidimensional)
    
#     return:
#     ---------
#     stats : dict
#         Dictionary containing the calculated statistics.
#     """
#     def maskChunked(arr, mask_fct, chunk_size=10_000_000):
#         lst = []
#         for start in range(0, arr.size, chunk_size):
#             end = min(start + chunk_size, arr.size)
#             chunk = arr.flat[start:end]
#             lst.append(chunk[mask_fct(chunk)])
#         vals = np.concatenate(lst)
#         return vals
    
#     # Convert input to numpy array and flatten
#     arr = np.array(array)
#     arr_flat = arr.flatten()
#     total_count = arr_flat.size

#     # Counting non-finite values
#     nan_count = np.isnan(arr_flat).sum()
#     pos_inf_count = np.isposinf(arr_flat).sum()
#     neg_inf_count = np.isneginf(arr_flat).sum()
#     non_finite_count = nan_count + pos_inf_count + neg_inf_count
#     finite_count = total_count - non_finite_count

#     # Rate calculs
#     if total_count > 0: 
#         nan_rate = nan_count / total_count
#         pos_inf_rate = pos_inf_count / total_count
#         neg_inf_rate = neg_inf_count / total_count
#     else:
#         nan_rate = pos_inf_rate = neg_inf_rate = np.nan
        
#     # Extraction of finite values
#     finite_vals = maskChunked(arr_flat, np.isfinite, chunk_size=10_000_000) 
    
#     # Calculation of additional statistics on finite values
#     if finite_count > 0:
#         min_val    = np.min(finite_vals)
#         q1         = np.percentile(finite_vals, 25)
#         median_val = np.median(finite_vals)
#         mean_val   = np.mean(finite_vals)
#         std_val    = np.std(finite_vals)
#         var_val    = np.var(finite_vals)
#         q3         = np.percentile(finite_vals, 75)
#         max_val    = np.max(finite_vals)
#         iqr        = q3 - q1
#         range_val  = max_val - min_val
#         unique_count = np.unique(finite_vals).size
#         zero_count = np.count_nonzero(finite_vals == 0)
#         zero_count_rate = zero_count / finite_count
    
#         skewness = skew(finite_vals)
#         kurt = kurtosis(finite_vals)

#     else:
#         min_val = q1 = median_val = mean_val = std_val = var_val = q3 = max_val = iqr = range_val = unique_count = zero_count = zero_count_rate = np.nan
#         skewness = kurt = np.nan

#     stats = {
#         'shape': arr.shape,
#         'total_count': total_count,
#         'count': int(finite_count),
#         'nan_count': int(nan_count),
#         'pos_inf_count': int(pos_inf_count),
#         'neg_inf_count': int(neg_inf_count),
#         'zero_count': int(zero_count) if not np.isnan(zero_count) else np.nan,
#         'nan_rate': nan_rate,
#         'pos_inf_rate': pos_inf_rate,
#         'neg_inf_rate': neg_inf_rate,
#         'zero_count_rate':zero_count_rate,
#         'min': min_val,
#         '25%': q1,
#         'median': median_val,
#         'mean': mean_val,
#         'std': std_val,
#         'var': var_val,
#         'IQR': iqr,
#         '75%': q3,
#         'max': max_val,
#         'range': range_val,
#         'unique_count': unique_count,
#         'skewness': skewness,
#         'kurtosis': kurt
#     }

#     return stats
def describeValues(array, chunk_size=1_000_000, sample_limit=5_000_000):
    """
    Memory-safe version of describeValues for very large arrays.
    Processes data in chunks instead of loading everything into memory.
    sample_limit is for calculate the median, quantil, etc. It become an estimation if superior to sample_limit
    """
    arr = np.asarray(array)
    shape = arr.shape
    total_count = arr.size
    
    # Check if array is numeric
    if np.issubdtype(arr.dtype, np.number):
        nan_count = pos_inf_count = neg_inf_count = zero_count = 0
        finite_count = 0

        # Initialize running stats
        finite_vals = []  
        finite_min = np.inf
        finite_max = -np.inf
        mean_accum = 0.0
        M2 = 0.0  # for variance (Welford’s method)
        
        # Iterate through array in chunks
        for start in range(0, total_count, chunk_size):
            end = min(start + chunk_size, total_count)
            chunk = arr.flat[start:end]

            # Count infinities / NaNs
            nan_mask = np.isnan(chunk)
            pos_mask = np.isposinf(chunk)
            neg_mask = np.isneginf(chunk)
            finite_mask = np.isfinite(chunk)

            nan_count += np.count_nonzero(nan_mask)
            pos_inf_count += np.count_nonzero(pos_mask)
            neg_inf_count += np.count_nonzero(neg_mask)

            finite_chunk = chunk[finite_mask]
            n = finite_chunk.size
            if n == 0:
                continue

            finite_count += n
            zero_count += np.count_nonzero(finite_chunk == 0)

            # Min / max
            finite_min = min(finite_min, np.min(finite_chunk))
            finite_max = max(finite_max, np.max(finite_chunk))

            # Incremental mean / variance (Welford)
            delta = finite_chunk.mean() - mean_accum
            mean_accum += delta * n / finite_count
            M2 += finite_chunk.var() * n  # variance sum approximation

            # Store chunk values for quantile/skew/kurtosis (optional)
            # if finite_count < 5_000_000:  # limit memory for quantiles
            #     finite_vals.append(finite_chunk)
            finite_vals.append(finite_chunk)

        # Combine finite values (or sample if too large)
        if finite_vals:
            if len(finite_vals) * chunk_size > sample_limit:
                last_rate = len(finite_vals[-1]) / chunk_size
                size_by_unit = sample_limit / (len(finite_vals) - 1 + last_rate)
                size_by_unit_int = int(size_by_unit)
                finite_vals = np.concatenate([np.random.choice(finite_vals[i], size_by_unit_int, replace=False) for i in range(len(finite_vals)-1)] + [np.random.choice(finite_vals[-1], int(size_by_unit*last_rate), replace=False)])
            else:
                finite_vals = np.concatenate(finite_vals)
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
            std_val = np.sqrt(M2 / finite_count)
            var_val = std_val**2
            q3 = np.percentile(finite_vals, 75)
            iqr = q3 - q1
            range_val = finite_max - finite_min
            unique_count = np.unique(finite_vals).size if finite_vals.size else np.nan
            zero_count_rate = zero_count / finite_count
            skewness = skew(finite_vals)
            kurt = kurtosis(finite_vals)
        else:
            finite_min = q1 = median_val = mean_val = std_val = var_val = q3 = finite_max = iqr = range_val = unique_count = zero_count_rate = np.nan
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
            'kurtosis': kurt
        }
    else:
        # String version
        unique_vals = Counter()
        max_len = 0
        min_len = np.inf

        for start in range(0, total_count, chunk_size):
            end = min(start + chunk_size, total_count)
            chunk = arr.flat[start:end]

            chunk = [str(x) for x in chunk]  # ensure all are strings
            lengths = [len(x) for x in chunk]
            max_len = max(max_len, max(lengths, default=0))
            min_len = min(min_len, min(lengths, default=0))
            unique_vals.update(chunk)

        most_common = unique_vals.most_common(5)
        stats =  {
            'shape': shape,
            'total_count': int(total_count),
            'unique_count': len(unique_vals),
            'min_length': min_len,
            'max_length': max_len,
            'most_common': most_common
        }

    return stats

def describe(array, min_threshold=1e-4, max_threshold=1e5, n_decimals=4):
    values = describeValues(array)
    for key, value in values.items():
        # Attempt to calculate the absolute value (for numerical cases)
        try:
            abs_val = abs(value)
        except TypeError:
            abs_val = None

        # Use scientific notation if the value is not zero and its magnitude
        # is less than min_threshold or greater than or equal to max_threshold
        if abs_val is not None and value != 0 and (abs_val < min_threshold or abs_val >= max_threshold):
            n = 2 if n_decimals is None else n_decimals
            formatted = f"{value:.{n}e}"                # value at scientific format
            if 'e' in formatted:
                parts = formatted.split("e")                # separate the mantissa and exponent
                mantissa = parts[0].rstrip("0").rstrip(".") # remove useless 0
                exponent = parts[1]                         
                print(f"{key}: {mantissa}e{exponent}")
            else:
                print(f"{key}: {formatted}")
        else:
            if n_decimals is not None and isinstance(value, (int, float)):
                print(f"{key}: {round(value, n_decimals)}")
            else:
                print(f"{key}: {value}")
                
def flatten(lst):
    result = []
    for l in lst:
        if isinstance(l, (list, tuple)):
            result.extend(flatten(l))  # Recursively flatten nested lists
        elif isinstance(l, np.ndarray):
            result.extend(l.flatten())  # Flatten NumPy arrays and extend the result list
        else:
            result.append(l)  # Append non-list and non-ndarray elements directly
    return result

def sort(lst):
    import re
    def naturalKey(s):
        """
        Transforms string s into a list of tokens:
        - each sequence of digits becomes a tuple (0, integer_value)
        - each sequence of non-numbers becomes a tuple (1, lowercase_string)
        In this way, all numeric blocks (0) come before letter blocks (1).
        """
        tokens = re.findall(r'\d+|\D+', s)
        key = []
        for tok in tokens:
            if tok.isdigit():
                key.append((0, int(tok)))
            else:
                key.append((1, tok.lower()))
        return key
    lst.sort(key=naturalKey)


def longuestConsecutiveSubArraySize(lst, sub_size=2):
    """
    :type fruits: List[int]
    :rtype: int
    """

    sub_unic = []
    total = 0
    total_cur = 0
    n_prev = 0

    for e in lst:
        # If the element is in the current pair current total +1
        if e in sub_unic:
            total_cur += 1

            # If it's not the last seen element put it as the last
            if e != sub_unic[-1]:
                e_remove = sub_unic.pop(sub_unic.index(e))
                sub_unic.append(e)
                n_prev = 1
            else:
                n_prev +=1
        # Else if the element is not in the current pair
        else:
            # And the pair array is at it's final size
            if len(sub_unic) == sub_size:
                total = max(total, total_cur)       # Update the total
                e_remove = sub_unic.pop(0)    # Remove the element from the previous pair
                total_cur = 1+n_prev
            else:
                total_cur += 1
                
            n_prev = 1
            sub_unic.append(e)

    return max(total, total_cur)