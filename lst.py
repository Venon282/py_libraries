import pickle
import warnings
import statistics
import joblib
import numpy as np
import math

def proportions(lst):
    return {str(element): list(lst).count(element) / len(lst) for element in set(lst)}

def countElements(lst):
    return {str(element): list(lst).count(element) for element in set(lst)}

def save(lst, path='./lst.pkl'):
    return joblib.dump(lst, path)

def load(path='./lst.pickle'):
    return joblib.load(path)

def interpolation(x_targ, x, y):
    """interpolation the y values for the x_targ values

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

def idxSumOfTwo(nums, target):
    """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
    """
    dict_ = {}
    for i in range(len(nums)):
        if nums[i] in dict_:
            return [dict_[nums[i]], i]
        dict_[target - nums[i]] = i
    return []
