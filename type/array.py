import numpy as np
from scipy.stats import trim_mean

def cropArray(arr: np.ndarray, newSize: int | tuple, method: str = 'left') -> np.ndarray:
    """
    Crop an array to the specified new size from the left, right, or middle.

    Args:
        arr: Input array to crop.
        newSize: Target size. If int, only crops the first dimension.
        method: Where to crop from:
            - 'l' or 'left' (default): crop from the beginning
            - 'r' or 'right': crop from the end
            - 'm' or 'middle': crop from the center

    Raises:
        ValueError: If method is invalid or array shape is smaller than newSize.

    Returns:
        np.ndarray: Cropped array.
    """
    if isinstance(newSize, int):
        newSize = (newSize,)
    
    if arr.ndim < len(newSize):
        raise ValueError(f"Array has {arr.ndim} dimensions, cannot crop to {len(newSize)} dimensions.")

    method = method.lower()
    if method not in ('left', 'l', 'right', 'r', 'middle', 'm'):
        raise ValueError(f"Invalid crop method: {method}. Choose from 'left', 'right', or 'middle'.")

    slices = []
    for i, size in enumerate(newSize):
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

    # Add full slices for any extra dimensions not specified in newSize
    slices.extend([slice(None)] * (arr.ndim - len(newSize)))
    
    return arr[tuple(slices)]

def circularMean(arr: np.ndarray, center: tuple[int, int] | None = None, threshold: float = np.sqrt(2)/2) -> np.ndarray:
    """Compute the mean of pixel values grouped by concentric rings around a center.

    Args:
        arr: Input 2D array.
        center: Center coordinates (y, x). If None, uses array center.
        threshold: Radius increment for ring grouping. Default is sqrt(2)/2.

    Returns:
        Array with the same shape as input, where each pixel value is replaced
        by the mean of its ring group.
    """
    nr, nc = arr.shape

    # Define the center if not provided
    if center is None:
        center = (nr//2, nc//2)
    
    y, x = np.indices((nr, nc)) # coordinates
    r = np.hypot(x - center[1], y - center[0]) # radial distance

    # Groups pixels by ring in a single vectorized pass
    bin_idx = np.round(r / threshold).astype(np.int64)
    sums = np.bincount(bin_idx.ravel(), weights=arr.ravel())
    counts = np.bincount(bin_idx.ravel())
    ring_mean = sums / counts

    return ring_mean[bin_idx]

def _groupedReduce(bin_idx, values, reduce_fn):
    """
    Applies a reduction function to each group defined by bin indices.

    Efficiently applies reduce_fn to each group defined by bin_idx, sorting only
    once (O(n log n)) rather than reconstructing an O(n_pixels) mask for each ring.

    Args:
        bin_idx (np.ndarray): Array of bin indices defining groups.
        values (np.ndarray): Array of values to reduce.
        reduce_fn (callable): Reduction function to apply to each group.

    Returns:
        np.ndarray: Array with the same shape as bin_idx, containing reduced values for each group.
    """
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

def circularMedian(arr: np.ndarray, center: tuple[int, int] | None = None, threshold: float = np.sqrt(2)/2) -> np.ndarray:
    """Compute the median of pixel values grouped by concentric rings around a center.

    Args:
        arr: Input 2D array.
        center: Center coordinates (y, x). If None, uses array center.
        threshold: Radius increment for ring grouping. Default is sqrt(2)/2.

    Returns:
        Array with the same shape as input, where each pixel value is replaced
        by the median of its ring group.
    """
    nr, nc = arr.shape

    # Define the center if not provided
    if center is None:
        center = (nr//2, nc//2)
    
    y, x = np.indices((nr, nc))
    r = np.hypot(x - center[1], y - center[0])
    bin_idx = np.round(r / threshold).astype(np.int64)
    return _groupedReduce(bin_idx, arr, np.median)



def circularTrimmedMean(arr: np.ndarray, center: tuple[int, int] | None = None, threshold: float = np.sqrt(2)/2, proportionToCut: float = 0.1) -> np.ndarray:
    """Compute the trimmed mean of pixel values grouped by concentric rings around a center.

    Args:
        arr: Input 2D array.
        center: Center coordinates (y, x). If None, uses array center.
        threshold: Radius increment for ring grouping. Default is sqrt(2)/2.
        proportionToCut: Proportion to trim from each end of the distribution.
            Default is 0.1 (10 percent from each end).

    Returns:
        Array with the same shape as input, where each pixel value is replaced
        by the trimmed mean of its ring group.
    """
    nr, nc = arr.shape
    if center is None:
        center = (nr // 2, nc // 2)

    y, x = np.indices((nr, nc))
    r = np.hypot(x - center[1], y - center[0])
    bin_idx = np.round(r / threshold).astype(np.int64)

    return _groupedReduce(bin_idx, arr, lambda seg: trim_mean(seg, proportionToCut))


def circularBilateral(arr: np.ndarray, center: tuple[int, int] | None = None, threshold: float = np.sqrt(2) / 2, sigmaR: float = 1.0, sigmaI: float = 0.1) -> np.ndarray:
    """Apply bilateral filtering to pixel values grouped by concentric rings around a center.

    Args:
        arr: Input 2D array.
        center: Center coordinates (y, x). If None, uses array center.
        threshold: Radius increment for ring grouping. Default is sqrt(2)/2.
        sigmaR: Spatial sigma parameter. Default is 1.0.
        sigmaI: Intensity sigma parameter for bilateral filter. Default is 0.1.

    Returns:
        Array with the same shape as input, where each pixel value is replaced
        by the bilateral filtered value of its ring group.
    """
    nr, nc = arr.shape
    if center is None:
        center = (nr // 2, nc // 2)

    y, x = np.indices((nr, nc))
    r = np.hypot(x - center[1], y - center[0])
    bin_idx = np.round(r / threshold).astype(np.int64)

    def _bilateralReduce(seg):
        i0 = seg.mean()
        wi = np.exp(-((seg - i0) ** 2) / (2 * sigmaI ** 2))
        return np.sum(wi * seg) / np.sum(wi)

    return _groupedReduce(bin_idx, arr, _bilateralReduce)

def gaussianMask(h: int, w: int, sigma: float = 0.4) -> np.ndarray:
    """Create a gaussian 2D centered mask with values between 0 and 1.

    Args:
        h: Height of the mask.
        w: Width of the mask.
        sigma: Standard deviation of the gaussian. Default is 0.4.

    Returns:
        2D gaussian mask with dtype np.float32, normalized to [0, 1].
    """
    y, x = np.ogrid[-1:1:h*1j, -1:1:w*1j]
    mask = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    mask = mask / mask.max()
    return mask.astype(np.float32)