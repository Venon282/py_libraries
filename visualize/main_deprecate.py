import os
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

# internal
from ..other.loggingUtils import getLogger
logger = getLogger(__name__)

def save(fig, file_path, dpi=200):
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    fig.savefig(file_path, dpi=dpi)

def _isFigureActive(ax: Axes) -> bool:
    fig_manager = ax.figure.canvas.manager                      # Get the figure's manager associated with the Axes
    active_fig_managers = plt._pylab_helpers.Gcf.figs.values()  # Get the list of currently active figure managers
    return fig_manager in active_fig_managers                    # If our figure manager is not in the active list, the figure was closed

    
def isOpen(ax: Axes) -> bool:
    return _isFigureActive(ax)

def isClose(ax: Axes) -> bool:
    return not _isFigureActive(ax)
    
def makeBins(data, bins=100, bin_type: str = "linear"):
    """
    Create histogram bin edges with different spacing types.

    Parameters
    -
    data : array-like
        The data from which to infer min/max values (used only if bins is int).
    bins : int or array-like
        - If int: defines the number of bins to generate.
        - If array-like: base edges to which the bin_type transformation is applied.
    bin_type : str, optional
        Type of bin spacing:
        - "linear" : no transformation (default)
        - "log" : logarithmic spacing
        - "sqrt" : square-root spacing
        - "quantile" : equal number of samples (ignored if bins are array-like)

    Returns
    -
    np.ndarray
        Array of bin edges.
    """
    data = np.asarray(data)
    data = data[np.isfinite(data)]  # remove NaN/Inf

    if data.size == 0:
        raise ValueError("Empty or invalid data array")

    bin_type = bin_type.lower()

    #  If bins are array-like, apply transform directly 
    if not isinstance(bins, int):
        bins = np.asarray(bins)
        if bin_type == "linear":
            return bins
        elif bin_type == "log":
            # interpret bins as linear positions, map to logspace range
            if np.any(bins <= 0):
                raise ValueError("Log binning requires positive bin edges")
            return np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        elif bin_type == "sqrt":
            if np.any(bins < 0):
                raise ValueError("Sqrt binning requires non-negative bin edges")
            return np.linspace(np.sqrt(bins[0]), np.sqrt(bins[-1]), len(bins)) ** 2
        else:
            raise ValueError(
                f"Unsupported bin_type '{bin_type}' for array-like bins. "
                "Supported: 'linear', 'log', 'sqrt'."
            )

    #  If bins is an integer, generate new edges 
    dmin, dmax = np.nanmin(data), np.nanmax(data)
    if dmin == dmax:
        dmin, dmax = dmin - 1, dmax + 1  # avoid degenerate case

    if bin_type == "linear":
        edges = np.linspace(dmin, dmax, bins)
    elif bin_type == "log":
        if dmin <= 0:
            dmin = np.min(data[data > 0]) if np.any(data > 0) else 1e-12
        edges = np.logspace(np.log10(dmin), np.log10(dmax), bins)
    elif bin_type == "sqrt":
        if dmin < 0:
            raise ValueError("Sqrt binning requires non-negative data")
        edges = np.linspace(np.sqrt(dmin), np.sqrt(dmax), bins) ** 2
    elif bin_type == "quantile":
        edges = np.quantile(data, np.linspace(0, 1, bins))
    else:
        raise ValueError(
            f"Unknown bin_type '{bin_type}'. Supported: 'linear', 'log', 'sqrt', 'quantile'"
        )

    return edges