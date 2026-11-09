"""Helpers to crop arrays, filter 2D arrays by concentric rings and build masks."""

from collections.abc import Callable
from typing import SupportsFloat

import numpy as np
from scipy.stats import trim_mean


def cropArray(
    arr: np.ndarray, new_size: int | tuple[int, ...], method: str = 'left'
) -> np.ndarray:
    """Crop an array to the specified new size from the left, right, or middle.

    With the `middle` method, when the number of removed elements of a dimension
    is odd, the extra element is removed at the end.

    Args:
        arr: Input array to crop.
        new_size: Target size. If int, only crops the first dimension. If a tuple,
            crops the first dimensions of `arr`, one size per dimension, and the
            remaining dimensions are kept whole.
        method: Where to crop from:
            - 'l' or 'left' (default): crop from the beginning
            - 'r' or 'right': crop from the end
            - 'm' or 'middle': crop from the center
            The value is case-insensitive.

    Returns:
        A view of `arr` with the cropped dimensions. No data is copied.

    Raises:
        ValueError: If `new_size` has more dimensions than `arr`, if `method` is
            invalid, or if a target size is larger than the matching dimension.
    """
    if isinstance(new_size, int):
        new_size = (new_size,)

    if arr.ndim < len(new_size):
        raise ValueError(
            f'Array has {arr.ndim} dimensions, '
            f'cannot crop to {len(new_size)} dimensions.'
        )

    method = method.lower()
    if method not in ('left', 'l', 'right', 'r', 'middle', 'm'):
        raise ValueError(
            f"Invalid crop method: {method}. Choose from 'left', 'right', or 'middle'."
        )

    slices: list[slice] = []
    for i, size in enumerate(new_size):
        max_size = arr.shape[i]
        if size > max_size:
            raise ValueError(
                f'Cannot crop dimension {i} to size {size}, '
                f'original size is {max_size}.'
            )

        if method in ('left', 'l'):
            start = 0
        elif method in ('right', 'r'):
            start = max_size - size
        else:
            start = (max_size - size) // 2

        slices.append(slice(start, start + size))

    # The dimensions after the last target size are kept whole
    slices.extend([slice(None)] * (arr.ndim - len(new_size)))

    return arr[tuple(slices)]


def _ringIndices(
    arr: np.ndarray, center: tuple[int, int] | None, threshold: float
) -> np.ndarray:
    """Assign each pixel of a 2D array to a concentric ring around a center.

    Args:
        arr: Input 2D array.
        center: Center coordinates (y, x). If None, uses the array center.
        threshold: Radius increment between two consecutive rings.

    Returns:
        Integer array with the shape of `arr`, holding the ring index of each pixel.

    Raises:
        ValueError: If `arr` is not 2D.
    """
    nr, nc = arr.shape

    if center is None:
        center = (nr // 2, nc // 2)

    y, x = np.indices((nr, nc))
    r = np.hypot(x - center[1], y - center[0])

    return np.round(r / threshold).astype(np.int64)


def circularMean(
    arr: np.ndarray,
    center: tuple[int, int] | None = None,
    threshold: float = np.sqrt(2) / 2,
) -> np.ndarray:
    """Compute the mean of pixel values grouped by concentric rings around a center.

    Args:
        arr: Input 2D array.
        center: Center coordinates (y, x). If None, uses array center.
        threshold: Radius increment between two consecutive rings. It must be
            positive. Default is sqrt(2)/2.

    Returns:
        Array with the same shape as input, where each pixel value is replaced
        by the mean of its ring group.

    Raises:
        ValueError: If `arr` is not 2D.
    """
    bin_idx = _ringIndices(arr, center, threshold)

    # Groups pixels by ring in a single vectorized pass
    flat_bins = bin_idx.ravel()
    sums = np.bincount(flat_bins, weights=arr.ravel())
    counts = np.bincount(flat_bins)

    # A ring index can be unused: dividing there would only raise a warning
    ring_mean: np.ndarray = np.divide(
        sums, counts, out=np.zeros_like(sums), where=counts > 0
    )

    smoothed: np.ndarray = ring_mean[bin_idx]
    return smoothed


def _groupedReduce(
    bin_idx: np.ndarray,
    values: np.ndarray,
    reduce_fn: Callable[[np.ndarray], SupportsFloat],
) -> np.ndarray:
    """Apply a reduction function to each group defined by bin indices.

    The values are sorted once (O(n log n)) rather than rebuilding an O(n_pixels)
    mask for each group.

    Args:
        bin_idx: Array of bin indices defining the groups.
        values: Array of values to reduce, with the same shape as `bin_idx`.
        reduce_fn: Function reducing the 1D array of the values of one group to a
            single number.

    Returns:
        Array with the shape of `bin_idx`, holding for each element the reduced
        value of its group.
    """
    flat_bins = bin_idx.ravel()
    flat_vals = values.ravel()
    order = np.argsort(flat_bins, kind='stable')
    sorted_bins = flat_bins[order]
    sorted_vals = flat_vals[order]

    _, start_idx = np.unique(sorted_bins, return_index=True)
    boundaries = np.append(start_idx, len(sorted_bins))

    out_sorted = np.empty(len(sorted_bins), dtype=float)
    for low, high in zip(boundaries[:-1], boundaries[1:], strict=True):
        out_sorted[low:high] = float(reduce_fn(sorted_vals[low:high]))

    out = np.empty_like(out_sorted)
    out[order] = out_sorted
    return out.reshape(bin_idx.shape)


def circularMedian(
    arr: np.ndarray,
    center: tuple[int, int] | None = None,
    threshold: float = np.sqrt(2) / 2,
) -> np.ndarray:
    """Compute the median of pixel values grouped by concentric rings around a center.

    Args:
        arr: Input 2D array.
        center: Center coordinates (y, x). If None, uses array center.
        threshold: Radius increment between two consecutive rings. It must be
            positive. Default is sqrt(2)/2.

    Returns:
        Array with the same shape as input, where each pixel value is replaced
        by the median of its ring group.

    Raises:
        ValueError: If `arr` is not 2D.
    """
    bin_idx = _ringIndices(arr, center, threshold)
    return _groupedReduce(bin_idx, arr, np.median)


def circularTrimmedMean(
    arr: np.ndarray,
    center: tuple[int, int] | None = None,
    threshold: float = np.sqrt(2) / 2,
    proportiontocut: float = 0.1,
) -> np.ndarray:
    """Compute the trimmed mean of pixel values grouped by concentric rings.

    Args:
        arr: Input 2D array.
        center: Center coordinates (y, x). If None, uses array center.
        threshold: Radius increment between two consecutive rings. It must be
            positive. Default is sqrt(2)/2.
        proportiontocut: Proportion to trim from each end of the distribution of a
            ring. Default is 0.1 (10 percent from each end). The name is the one of
            the matching argument of `scipy.stats.trim_mean`. A ring with fewer
            than 1 / proportiontocut pixels is not trimmed. A value of 0.5 or more
            is not meaningful: it can trim a ring down to nothing, which gives NaN.

    Returns:
        Array with the same shape as input, where each pixel value is replaced
        by the trimmed mean of its ring group.

    Raises:
        ValueError: If `arr` is not 2D, or if `proportiontocut` is too big for a
            ring, as `scipy.stats.trim_mean` raises it (above 0.5).
    """
    bin_idx = _ringIndices(arr, center, threshold)
    return _groupedReduce(
        bin_idx,
        arr,
        lambda segment: trim_mean(segment, proportiontocut=proportiontocut),
    )


def circularBilateral(
    arr: np.ndarray,
    center: tuple[int, int] | None = None,
    threshold: float = np.sqrt(2) / 2,
    sigma_r: float = 1.0,
    sigma_i: float = 0.1,
) -> np.ndarray:
    """Apply bilateral filtering to pixel values grouped by concentric rings.

    Inside a ring, each pixel is weighted by a Gaussian of its distance to the mean
    of the ring, so the pixels far from the mean (outliers) weigh little.

    Args:
        arr: Input 2D array.
        center: Center coordinates (y, x). If None, uses array center.
        threshold: Radius increment between two consecutive rings. It must be
            positive. Default is sqrt(2)/2.
        sigma_r: Spatial sigma parameter. It is currently unused: the weights only
            depend on the intensity.
        sigma_i: Intensity sigma parameter of the filter. It must be positive, and
            a value much lower than the spread of a ring can give NaN.

    Returns:
        Array with the same shape as input, where each pixel value is replaced
        by the bilateral filtered value of its ring group.

    Raises:
        ValueError: If `arr` is not 2D.
    """
    bin_idx = _ringIndices(arr, center, threshold)

    def bilateralReduce(segment: np.ndarray) -> float:
        i0 = segment.mean()
        wi = np.exp(-((segment - i0) ** 2) / (2 * sigma_i**2))
        return float(np.sum(wi * segment) / np.sum(wi))

    return _groupedReduce(bin_idx, arr, bilateralReduce)


def gaussianMask(h: int, w: int, sigma: float = 0.4) -> np.ndarray:
    """Create a gaussian 2D centered mask with values between 0 and 1.

    The coordinates are normalized to [-1, 1] on each axis, so `sigma` is relative
    to the half size of the mask. The mask is divided by its maximum, so its
    maximum is exactly 1, even when the size is even and no pixel is at the center.

    Args:
        h: Height of the mask.
        w: Width of the mask.
        sigma: Standard deviation of the gaussian, in normalized coordinates. It
            must be positive. Default is 0.4.

    Returns:
        2D gaussian mask of shape `(h, w)` with dtype np.float32, normalized to
        [0, 1].

    Raises:
        ValueError: If `h` or `w` is 0, because the mask is then empty.
    """
    y, x = np.ogrid[-1 : 1 : h * 1j, -1 : 1 : w * 1j]
    mask: np.ndarray = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    normalized: np.ndarray = mask / mask.max()
    return normalized.astype(np.float32)
