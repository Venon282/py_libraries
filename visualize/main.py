import os
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ..other.loggingUtils import getLogger

logger = getLogger(__name__)


def save(fig, file_path: str, dpi: int = 150) -> None:
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    fig.savefig(file_path, dpi=dpi)


def _isFigureActive(ax: Axes) -> bool:
    fig_manager = ax.figure.canvas.manager
    active_fig_managers = plt._pylab_helpers.Gcf.figs.values()
    return fig_manager in active_fig_managers


def isOpen(ax: Axes) -> bool:
    return _isFigureActive(ax)


def isClose(ax: Axes) -> bool:
    return not _isFigureActive(ax)


def _unwrap(arg: Any) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Parse a plot argument into (x, y, opts).

    Accepted forms:
      - y_array
      - (y_array, opts_dict)
      - (x_array, y_array)
      - (x_array, y_array, opts_dict)
    """
    opts: Dict[str, Any] = {}
    if isinstance(arg, (list, np.ndarray)):
        y = np.asarray(arg)
        x = np.arange(len(y))
    elif isinstance(arg, tuple) and len(arg) in (2, 3):
        first, second = arg[0], arg[1]
        if isinstance(second, dict):
            y, opts = np.asarray(first), dict(second)
            x = np.arange(len(y))
        else:
            x, y = np.asarray(first), np.asarray(second)
            if len(arg) == 3:
                opts = dict(arg[2])
    else:
        raise ValueError(f"Cannot parse plot argument: {arg}")
    return x, y, opts


def makeBins(data, bins: int = 100, bin_type: str = "linear") -> np.ndarray:
    """
    Create histogram bin edges with different spacing types.

    Parameters
    ----------
    data : array-like
        Source data; used to infer range when bins is an integer.
    bins : int or array-like
        Number of bins (int) or explicit base edges (array-like).
    bin_type : str, optional
        Spacing strategy: 'linear' (default), 'log', 'sqrt', 'quantile'.
        'quantile' is only valid when bins is an integer.

    Returns
    -------
    np.ndarray
        Array of bin edges.
    """
    data = np.asarray(data)
    data = data[np.isfinite(data)]

    if data.size == 0:
        raise ValueError("Empty or invalid data array")

    bin_type = bin_type.lower()

    if not isinstance(bins, int):
        bins = np.asarray(bins)
        if bin_type == "linear":
            return bins
        if bin_type == "log":
            if np.any(bins <= 0):
                raise ValueError("Log binning requires positive bin edges")
            return np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        if bin_type == "sqrt":
            if np.any(bins < 0):
                raise ValueError("Sqrt binning requires non-negative bin edges")
            return np.linspace(np.sqrt(bins[0]), np.sqrt(bins[-1]), len(bins)) ** 2
        raise ValueError(
            f"Unsupported bin_type '{bin_type}' for array-like bins. "
            "Supported: 'linear', 'log', 'sqrt'."
        )

    dmin, dmax = np.nanmin(data), np.nanmax(data)
    if dmin == dmax:
        dmin, dmax = dmin - 1, dmax + 1

    if bin_type == "linear":
        return np.linspace(dmin, dmax, bins)
    if bin_type == "log":
        if dmin <= 0:
            dmin = np.min(data[data > 0]) if np.any(data > 0) else 1e-12
        return np.logspace(np.log10(dmin), np.log10(dmax), bins)
    if bin_type == "sqrt":
        if dmin < 0:
            raise ValueError("Sqrt binning requires non-negative data")
        return np.linspace(np.sqrt(dmin), np.sqrt(dmax), bins) ** 2
    if bin_type == "quantile":
        return np.quantile(data, np.linspace(0, 1, bins))
    raise ValueError(
        f"Unknown bin_type '{bin_type}'. Supported: 'linear', 'log', 'sqrt', 'quantile'"
    )