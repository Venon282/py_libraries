import numpy as np

def crop(arr: np.ndarray, new_size: int | tuple, method: str = 'left') -> np.ndarray:
    """
    Crop an array to the specified new size from the left, right, or middle.

    Args:
        arr (np.ndarray): Input array to crop.
        new_size (int | tuple): Target size. If int, only crops the first dimension.
        method (str, optional): Where to crop from:
            - 'l' or 'left' (default): crop from the beginning
            - 'r' or 'right': crop from the end
            - 'm' or 'middle': crop from the center

    Raises:
        ValueError: If method is invalid or array shape is smaller than new_size.

    Returns:
        np.ndarray: Cropped array.
    """
    if isinstance(new_size, int):
        new_size = (new_size,)
    
    if arr.ndim < len(new_size):
        raise ValueError(f"Array has {arr.ndim} dimensions, cannot crop to {len(new_size)} dimensions.")

    method = method.lower()
    if method not in ('left', 'l', 'right', 'r', 'middle', 'm'):
        raise ValueError(f"Invalid crop method: {method}. Choose from 'left', 'right', or 'middle'.")

    slices = []
    for i, size in enumerate(new_size):
        max_size = arr.shape[i]
        if size > max_size:
            raise ValueError(f"Cannot crop dimension {i} to size {size}, original size is {max_size}.")

        if method in ('left', 'l'):
            start = 0
        elif method in ('right', 'r'):
            start = max_size - size
        elif method in ('middle', 'm'):
            start = (max_size - size) // 2

        end = start + size
        slices.append(slice(start, end))

    # Add full slices for any extra dimensions not specified in new_size
    slices.extend([slice(None)] * (arr.ndim - len(new_size)))
    
    return arr[tuple(slices)]

def circularMean(arr, center=None, threshold = np.sqrt(2)/2):
    nr, nc = arr.shape

    # Define the center if not provided
    if center is None:
        center = (nr//2, nc//2)
    
    y, x = np.indices((nr, nc)) # coordonnées
    r = np.hypot(x - center[1], y - center[0]) # distance radiale

    # Groups pixels by ring in a single vectorized pass
    bin_idx = np.round(r / threshold).astype(np.int64)
    sums = np.bincount(bin_idx.ravel(), weights=arr.ravel())
    counts = np.bincount(bin_idx.ravel())
    ring_mean = sums / counts

    return ring_mean[bin_idx]

def _groupedReduce(bin_idx, values, reduce_fn):
    """Applies `reduce_fn` to each group defined by `bin_idx`, sorting only
    once (O(n log n)) rather than reconstructing an O(n_pixels) mask for each ring."""
    flat_bins = bin_idx.ravel()
    flat_vals = values.ravel()
    order = np.argsort(flat_bins, kind='stable')
    sorted_bins = flat_bins[order]
    sorted_vals = flat_vals[order]

    unique_bins, start_idx = np.unique(sorted_bins, return_index=True)
    boundaries = np.append(start_idx, len(sorted_bins))

    out_sorted = np.empty(len(sorted_bins), dtype=float)
    for i in range(len(unique_bins)):
        segment = sorted_vals[boundaries[i]:boundaries[i + 1]]
        out_sorted[boundaries[i]:boundaries[i + 1]] = reduce_fn(segment)

    out = np.empty_like(out_sorted)
    out[order] = out_sorted
    return out.reshape(bin_idx.shape)

def circularMedian(arr, center=None, threshold = np.sqrt(2)/2):
    nr, nc = arr.shape

    # Define the center if not provided
    if center is None:
        center = (nr//2, nc//2)
    
    y, x = np.indices((nr, nc))
    r = np.hypot(x - center[1], y - center[0])
    bin_idx = np.round(r / threshold).astype(np.int64)
    return _groupedReduce(bin_idx, arr, np.median)

from scipy.stats import trim_mean

def circularTrimmedMean(arr, center=None, threshold=np.sqrt(2)/2, proportiontocut=0.1):
    nr, nc = arr.shape
    new_arr = np.zeros_like(arr, dtype=float)
    if center is None:
        center = (nr//2, nc//2)
    y, x = np.indices((nr, nc))
    r = np.hypot(x - center[1], y - center[0])
    for unic in np.unique(r):
        mask = np.abs(r - unic) < threshold
        vals = arr[mask]
        new_arr[mask] = trim_mean(vals, proportiontocut)
    return new_arr


def circularBilateral(arr, center=None, threshold=np.sqrt(2)/2,
                      sigma_r=1.0, sigma_i=0.1):
    nr, nc = arr.shape
    new_arr = np.zeros_like(arr, dtype=float)
    if center is None:
        center = (nr//2, nc//2)
    y, x = np.indices((nr, nc))
    r = np.hypot(x - center[1], y - center[0])
    for unic in np.unique(r):
        mask = np.abs(r - unic) < threshold
        vals = arr[mask]
        d = np.abs(r[mask] - unic)
        # poids radiaux
        wr = np.exp(-(d**2)/(2*sigma_r**2))
        # poids intensité (par rapport à la moyenne)
        i0 = vals.mean()
        wi = np.exp(-((vals - i0)**2)/(2*sigma_i**2))
        w = wr * wi
        new_arr[mask] = np.sum(w * vals) / np.sum(w)
    return new_arr

def gaussianMask(h, w, sigma=0.4):
    """Create a gaussian  2D centred mask (values between 0 and 1)."""
    y, x = np.ogrid[-1:1:h*1j, -1:1:w*1j]
    mask = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    mask = mask / mask.max()
    return mask.astype(np.float32)