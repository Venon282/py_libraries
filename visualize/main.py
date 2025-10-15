from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def save(fig, file_path, parents=True, exist_ok=True):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=parents, exist_ok=exist_ok)
    file_path.touch(exist_ok=exist_ok)
    fig.savefig(str(file_path)) 
    
def isClose(ax: Axes) -> bool:
    """
    Check whether the Matplotlib figure window associated with a given Axes object has been closed.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object whose parent figure should be checked.
    Returns
    -------
    bool
        True if the figure window has been closed, False otherwise.
    """
    fig_manager = ax.figure.canvas.manager # Get the figure's manager associated with the Axes
    active_fig_managers = plt._pylab_helpers.Gcf.figs.values() # Get the list of currently active figure managers
    return fig_manager not in active_fig_managers # If our figure manager is not in the active list, the figure was closed

def isOpen(ax: Axes) -> bool:
    """
    Check whether the Matplotlib figure window associated with a given Axes object has is still open.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object whose parent figure should be checked.
    Returns
    -------
    bool
        True if the figure window has been closed, False otherwise.
    """
    fig_manager = ax.figure.canvas.manager # Get the figure's manager associated with the Axes
    active_fig_managers = plt._pylab_helpers.Gcf.figs.values() # Get the list of currently active figure managers
    return fig_manager in active_fig_managers # If our figure manager is not in the active list, the figure was closed

def _createPlot(**kwargs):
    fig = kwargs.pop('fig', None)
    ax = kwargs.pop('ax', None)
    if fig is None and ax is None:
        return plt.subplots(figsize=(10, 6))
    elif fig is None:
        return ax.figure, ax
    elif ax is None:
        print('WARNING: you provided the fig, but not the ax.')
        return fig, fig.axes[-1]
    else:
        return fig, ax