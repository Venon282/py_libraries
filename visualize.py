import matplotlib.pyplot as plt
import plotly.express as px
import math
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.lines as mlines
from matplotlib.patches import Patch
from plotly.validator_cache import ValidatorCache

import scipy.stats as scipy_stats
from typing import Any, Dict, List, Optional, Tuple, Union




from matplotlib import cm, colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

class Visualize:
    class Maker:
        @staticmethod
        def _all_markers():
            """
            Retrieve the full raw list of all Plotly marker symbols (strings only).
            In this Plotly build, the symbol name lives at raw[i+2] for each chunk of 3.
            """
            raw = ValidatorCache.get_validator("scatter.marker", "symbol").values
            names = []
            # values is [something, something, name, something, something, name, …]
            for i in range(0, len(raw), 3):
                candidate = raw[i + 2]
                names.append(candidate)
            return names

        @staticmethod
        def _filter(predicate):
            """Helper: filter the full marker list by a predicate on the name."""
            return [m for m in Visualize.Maker._all_markers() if predicate(m)]

        @staticmethod
        def simple():
            """Markers without any suffix or prefix (e.g. 'circle', 'square')."""
            return Visualize.Maker._filter(lambda m: '-' not in m)

        @staticmethod
        def open():
            """Markers with '-open' suffix, excluding '-open-dot'."""
            return Visualize.Maker._filter(lambda m: m.endswith('-open') and not m.endswith('-open-dot'))

        @staticmethod
        def dot():
            """Markers with '-dot' suffix, excluding '-open-dot'."""
            return Visualize.Maker._filter(lambda m: m.endswith('-dot') and not m.endswith('-open-dot'))

        @staticmethod
        def open_dot():
            """Markers with '-open-dot' suffix."""
            return Visualize.Maker._filter(lambda m: m.endswith('-open-dot'))

        @staticmethod
        def triangle():
            """All triangle markers (all orientations)."""
            return Visualize.Maker._filter(lambda m: m.startswith('triangle'))

        @staticmethod
        def polygon():
            """Polygon markers: pentagon, hexagon, octagon, etc."""
            prefixes = ('pentagon', 'hexagon', 'octagon')
            return Visualize.Maker._filter(lambda m: any(m.startswith(p) for p in prefixes))

        @staticmethod
        def star():
            """Star-shaped markers (including hexagram)."""
            return Visualize.Maker._filter(lambda m: 'star' in m or 'hexagram' in m)

        @staticmethod
        def cross():
            """Cross markers, excluding circle-cross/square-cross variants."""
            return Visualize.Maker._filter(
                lambda m: 'cross' in m and not m.startswith(('circle-cross', 'square-cross'))
            )

        @staticmethod
        def x():
            """X-shaped markers (including 'x' itself)."""
            return Visualize.Maker._filter(lambda m: m == 'x' or m.endswith('-x'))

        @staticmethod
        def arrow():
            """Simple arrow markers (no bar)."""
            return Visualize.Maker._filter(lambda m: m.startswith('arrow-') and not m.startswith('arrow-bar-'))

        @staticmethod
        def arrow_bar():
            """Arrow markers with a bar (e.g. 'arrow-bar-up')."""
            return Visualize.Maker._filter(lambda m: m.startswith('arrow-bar-'))

        @staticmethod
        def line():
            """Line markers (EW, NS, NE, NW, etc.)."""
            return Visualize.Maker._filter(lambda m: m.startswith('line-'))

        @staticmethod
        def y():
            """Y-shaped markers."""
            return Visualize.Maker._filter(lambda m: m.startswith('y-'))

        @staticmethod
        def other():
            """All markers not covered by the above categories."""
            cats = set(
                Visualize.Maker.simple() +
                Visualize.Maker.open() +
                Visualize.Maker.dot() +
                Visualize.Maker.open_dot() +
                Visualize.Maker.triangle() +
                Visualize.Maker.polygon() +
                Visualize.Maker.star() +
                Visualize.Maker.cross() +
                Visualize.Maker.x() +
                Visualize.Maker.arrow() +
                Visualize.Maker.arrow_bar() +
                Visualize.Maker.line() +
                Visualize.Maker.y()
            )
            return [m for m in Visualize.Maker._all_markers() if m not in cats]

        @staticmethod
        def all():
            """Return the complete list of marker symbols."""
            return Visualize.Maker._all_markers()
        
    class Cmap:
        @staticmethod
        def _get_scales(module):
            """
            Retrieve all list-attributes of a colors module (e.g. px.colors.sequential),
            and separate normal and '_r' versions.
            """
            all_attrs = [name for name in dir(module) if not name.startswith('_')]
            normal = [
                name.lower()
                for name in all_attrs
                if isinstance(getattr(module, name), list) and not name.endswith('_r')
            ]
            reversed_ = [
                name.lower()
                for name in all_attrs
                if isinstance(getattr(module, name), list) and name.endswith('_r')
            ]
            return normal, reversed_

        @staticmethod
        def cmaps():
            """All named colorscales (with both normal and reversed names)."""
            return px.colors.named_colorscales()

        # Sequential (single-hue or multi-hue)
        @staticmethod
        def sequential():
            normal, _ = Visualize.Cmap._get_scales(px.colors.sequential)
            return normal

        @staticmethod
        def sequential_r():
            _, rev = Visualize.Cmap._get_scales(px.colors.sequential)
            return rev

        # Diverging
        @staticmethod
        def diverging():
            normal, _ = Visualize.Cmap._get_scales(px.colors.diverging)
            return normal

        @staticmethod
        def diverging_r():
            _, rev = Visualize.Cmap._get_scales(px.colors.diverging)
            return rev

        # Cyclical
        @staticmethod
        def cyclical():
            normal, _ = Visualize.Cmap._get_scales(px.colors.cyclical)
            return normal

        @staticmethod
        def cyclical_r():
            _, rev = Visualize.Cmap._get_scales(px.colors.cyclical)
            return rev

        # Qualitative (discrete/category)
        @staticmethod
        def qualitative():
            normal, _ = Visualize.Cmap._get_scales(px.colors.qualitative)
            return normal

        @staticmethod
        def qualitative_r():
            _, rev = Visualize.Cmap._get_scales(px.colors.qualitative)
            return rev

    @staticmethod
    def save(fig, file_path, parents=True, exist_ok=True):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=parents, exist_ok=exist_ok)
        file_path.touch(exist_ok=exist_ok)
        fig.savefig(str(file_path)) 
    
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

        
    class Plot:
        """Unified plotting interface. All methods accept:
           - *args: each can be
               • y-array
               • (x, y)
               • (x, y, {opts})
           - fig, ax: pass existing fig/ax if desired.
           - path (opt), show (bool)
           - global styling kwargs: title, xlabel, ylabel, grid, xlim, ylim, legend, xscale, yscale
        """

        @staticmethod
        def _init(kwargs: Dict[str, Any]) -> Tuple[Figure, Axes, Dict[str, Any], Dict[str, Any]]:
            fig = kwargs.pop('fig', None)
            ax  = kwargs.pop('ax', None)
            if ax is None:
                fig, ax = plt.subplots(figsize=kwargs.pop('figsize',(10,6)))
            elif fig is None:
                fig = ax.figure

            # extract file/show
            path = kwargs.pop('path', None)
            show = kwargs.pop('show', True)

            # styling
            style = {
                'title':  kwargs.pop('title', ''),
                'xlabel': kwargs.pop('xlabel', ''),
                'ylabel': kwargs.pop('ylabel', ''),
                'grid':   kwargs.pop('grid', True),
            }
            extras = {
                'xlim':  kwargs.pop('xlim', None),
                'ylim':  kwargs.pop('ylim', None),
                'xscale':kwargs.pop('xscale', None),
                'yscale':kwargs.pop('yscale', None),
                'legend':kwargs.pop('legend', False),
            }
            return fig, ax, style, extras, path, show

        @staticmethod
        def _apply_style(fig: Figure, ax: Axes,
                         style: Dict[str, Any], extras: Dict[str, Any]) -> None:
            ax.set_title(style['title'])
            ax.set_xlabel(style['xlabel'])
            ax.set_ylabel(style['ylabel'])
            ax.grid(style['grid'])
            if extras['xlim'] is not None:   ax.set_xlim(extras['xlim'])
            if extras['ylim'] is not None:   ax.set_ylim(extras['ylim'])
            if extras['xscale'] is not None: ax.set_xscale(extras['xscale'])
            if extras['yscale'] is not None: ax.set_yscale(extras['yscale'])
            if extras['legend']:
                ax.legend()
            fig.tight_layout()

        @staticmethod
        def _unwrap(arg: Any) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
            """Support y-only, (x,y), or (x,y,opts)."""
            opts: Dict[str, Any] = {}
            if isinstance(arg, (list, np.ndarray)):
                y = np.asarray(arg)
                x = np.arange(len(y))
            elif isinstance(arg, tuple) and len(arg) in (2,3):
                x, y = arg[0], arg[1]
                if len(arg)==3:
                    opts = dict(arg[2])
                x = np.asarray(x); y = np.asarray(y)
            else:
                raise ValueError(f"Can't parse argument {arg}")
            return x, y, opts
        
        @staticmethod
        def _end(fig, ax, style, extras, path, show):
            Visualize.Plot._apply_style(fig, ax, style, extras)
            if path:    Visualize.save(fig, path)
            if show:    plt.show(); return None
            return fig

        @staticmethod
        def curve(*args, **kwargs) -> Optional[Figure]:
            """Simple line plot."""
            fig, ax, style, extras, path, show = Visualize.Plot._init(kwargs)
            for arg in args:
                x, y, opts = Visualize.Plot._unwrap(arg)
                ax.plot(x, y, **opts)
            return Visualize.Plot._end(fig, ax, style, extras, path, show)


        @staticmethod
        def scatter(*args, **kwargs) -> Optional[Figure]:
            """Scatter / dot plot."""
            fig, ax, style, extras, path, show = Visualize.Plot._init(kwargs)
            for arg in args:
                x, y, opts = Visualize.Plot._unwrap(arg)
                ax.scatter(x, y, **opts)
            return Visualize.Plot._end(fig, ax, style, extras, path, show)
        
        @staticmethod
        def scatter_density(*args, bw_method: Optional[Union[str, float]] = None, cbar_label: Optional[str] = None, **kwargs) -> Optional[Figure]:
            """
            Scatter plot colored by a 2D Gaussian KDE estimate of density.
            
            Parameters:
            - *args: each of
                • y-array
                • (x, y)
                • (x, y, {opts})
            - bw_method: passed to gaussian_kde (None for scott's rule, or float/string)
            - cbar_label: label for the density colorbar. If none no color bar
            - plus all the usual kwargs (fig, ax, title, xlabel, path, show, etc.)
            """
            fig, ax, style, extras, path, show = Visualize.Plot._init(kwargs)

            for arg in args:
                x, y, opts = Visualize.Plot._unwrap(arg)

                # compute 2D KDE
                xy = np.vstack([x, y])
                kde = scipy_stats.gaussian_kde(xy, bw_method=bw_method)
                z = kde(xy)

                # sort the points by density, so that high-density points are on top
                idx = z.argsort()

                sc = ax.scatter(x[idx], y[idx],c=z[idx],**opts)

            # add a colorbar for density
            if cbar_label is not None:
                fig.colorbar(sc, ax=ax, label=cbar_label)

            return Visualize.Plot._end(fig, ax, style, extras, path, show)

        @staticmethod
        def histogram(data: Union[List, np.ndarray], bins: int=30, **kwargs) -> Optional[Figure]:
            """Univariate histogram."""
            fig, ax, style, extras, path, show = Visualize.Plot._init(kwargs)
            ax.hist(data, bins=bins, **kwargs.pop('histopts', {}))
            return Visualize.Plot._end(fig, ax, style, extras, path, show)

        @staticmethod
        def heatmap(*args, bins: int=100, cmap: str='viridis', **kwargs) -> Optional[Figure]:
            """2D density heatmap."""
            fig, ax, style, extras, path, show = Visualize.Plot._init(kwargs)
            for arg in args:
                x, y, opts = Visualize.Plot._unwrap(arg)
                heat, xedges, yedges = np.histogram2d(x, y, bins=bins)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                im = ax.imshow(
                    heat.T, extent=extent, origin='lower',
                    aspect='auto', cmap=cm.get_cmap(cmap),
                    **opts
                )
                fig.colorbar(im, ax=ax, label=kwargs.get('cbar_label',''))
            return Visualize.Plot._end(fig, ax, style, extras, path, show)

        @staticmethod
        def candles(data: np.ndarray,
                    *,
                    width: float=0.6,
                    open_color: str='green', close_color: str='red',
                    wick_width: float=1.0,
                    legend: bool=True,
                    **kwargs) -> Optional[Figure]:
            """
            OHLC candlestick chart.
            `data` is an (N,4) array of [open, high, low, close].
            """
            fig, ax, style, extras, path, show = Visualize.Plot._init(kwargs)
            data = np.asarray(data)
            xs = np.arange(len(data))
            for i, (o, h, l, c) in enumerate(data):
                color = open_color if c >= o else close_color
                # wick
                ax.plot([xs[i], xs[i]], [l, h], color=color, lw=wick_width)
                # body
                low, high = sorted((o, c))
                rect = Patch((xs[i]-width/2, low), width, high-low,
                             facecolor=color, edgecolor=color)
                ax.add_patch(rect)
            if legend:
                ax.add_patch(Patch(facecolor=open_color, label='Up'))
                ax.add_patch(Patch(facecolor=close_color, label='Down'))
                ax.legend()
            return Visualize.Plot._end(fig, ax, style, extras, path, show)
            
        
# ------------------------------------------
# GENERAL FUNCTIONS
# ------------------------------------------

def save(fig, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch(exist_ok=True)
    fig.savefig(str(file_path))
    




    
def _end(fig, ax, kwargs2, path, show):
    """
    Returns:
        plt: if don't want to save or show
        None: if show or save
    """
    _kwargs2(fig, ax, kwargs2)
    
    fig.tight_layout()
    # Save the plot if a path is provided
    if path is not None:
        save(fig, file_path=path) # plt.savefig(path)
    
    if show:
        plt.show()
    elif path is None:
        return fig    
    else:
        plt.close(fig)
    
def _normalise(c):
    c = np.array(c)
    c_min = np.min(c)
    c_max = np.max(c)
    if c_max != c_min:
        return (c - c_min) / (c_max - c_min)
    else:
        # For a constant curve, use a middle value
        return np.full_like(c, 0.5)
    
def _arg2d(arg):
    options = {}
    # Support for (x, y, options) or (y, options) input formats
    if isinstance(arg, (tuple, list)):
        if len(arg) == 3:
            if isinstance(arg[2], dict):
                options.update(arg[2])
                x, y = arg[0], arg[1]
            else: raise AttributeError('Third argument in args not handle')
        elif len(arg) == 2:
            if isinstance(arg[1], dict):
                options.update(arg[1])
                y = arg[0]
                x = np.arange(len(y))
            else:
                x, y = arg[0], arg[1]
        elif len(arg) == 1:
            x , y = np.arange(len(arg[0])), arg[0]
        else: raise AttributeError(f'Dimension {len(arg)} not handle')
    else:
        x, y = np.arange(len(arg)), arg

    return x, y, options

def _kwargs2(fig, ax, kwargs):
    

    if kwargs.pop('legend_fig', False) or kwargs.pop('legend', False):
        fig.legend()
    elif kwargs.pop('legend_ax', False):
        ax.legend()

    _apply(kwargs, 'xscale', ax.set_xscale)
    _apply(kwargs, 'yscale', ax.set_yscale)
    
    ax.relim()          # Recalcule les limites des données
    ax.autoscale_view() # Applique ces limites à la vue
        
def _apply(kwargs, label, function):
    element =  kwargs.pop(label, False)
    if element is not False:
        function(element)
    
def _kwargs(ax, kwargs):
    
            
    ax.set_title(kwargs.pop('title', ''))
    ax.grid(kwargs.pop('grid', None))
    ax.set_xlabel(kwargs.pop('xlabel', None))
    ax.set_ylabel(kwargs.pop('ylabel', None))
    
    _apply(kwargs, 'ylim', ax.set_ylim)
    _apply(kwargs, 'xlim', ax.set_xlim)
        
    kwargs2 = {'legend': kwargs.pop('legend', False), 'legend_ax': kwargs.pop('legend_ax', False), 'legend_fig': kwargs.pop('legend_fig', False), 'xscale': kwargs.pop('xscale', False), 'yscale': kwargs.pop('yscale', False)}

    return kwargs, kwargs2

def _additionalLines(ax, additional_lines, global_options):
    if len(additional_lines) == 0:
        return
    
    # Get the list of all valid keyword arguments for Line2D
    dummy_line = mlines.Line2D([], [])  # Create a dummy Line2D object to access its properties
    valid_props = set(dummy_line.properties().keys())  # returns a set of valid property names

    for al in additional_lines:
        options = global_options.copy()
        x, y, new_options = _arg2d(al)
        options.update(new_options)
        # Filter out options that are not valid for Line2D
        valid_options = {key: value for key, value in options.items() if key in valid_props}
        ax.plot(x, y, **valid_options)

# ------------------------------------------
# PLOTS
# ------------------------------------------

# def curve(x, y, **kwargs):
#     """
#     Plots a simple curve with customization options.
    
#     Parameters:
#     - x: The x-values (list or array-like).
#     - y: The y-values (list or array-like).
#     - title: The title of the plot (optional).
#     - xlabel: Label for the x-axis (optional).
#     - ylabel: Label for the y-axis (optional).
#     - color: Color of the line (default is 'blue').
#     - linestyle: Line style (default is solid line '-').
#     - linewidth: Width of the line (default is 2).
#     - grid: Whether to show the grid (default is True).
#     """
#     plt.figure(figsize=(10, 6)) 
    
#     # options, kwargs2 = _kwargs(plt, kwargs)
#     path, show = kwargs.pop('path', None), kwargs.pop('show', True)
    
#     plt.plot(x, y, options)
    
#     # _kwargs2(plt, kwargs2)
    
#     return _end(fig, path, show)
    


def curves(*args, **kwargs):
    """
    Plots one or more curves. Supports coloring with a colormap (`c`) and plotting on a provided figure/axes.
    """
    fig = kwargs.pop('fig', None)
    ax = kwargs.pop('ax', None)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))  # create new if none provided

    global_options, kwargs2 = _kwargs(ax, kwargs)

    path, show = kwargs.pop('path', None), kwargs.pop('show', True)
    normalize_x, normalize_y = kwargs.pop('normalize_x', False), kwargs.pop('normalize_y', False)
    c = kwargs.pop('c', False)

    if c is not False:
        if kwargs.pop('c_scale', 'linear') == 'log':
            norm = mcolors.LogNorm(vmin=min(c), vmax=max(c))
        else:
            norm = mcolors.Normalize(vmin=min(c), vmax=max(c))
        cmap = cm.get_cmap(kwargs.pop('cmap', 'viridis'))
        colors = [cmap(norm(element)) for element in c]
        c_label = kwargs.pop('c_label', None)

    for i, arg in enumerate(args):
        options = global_options.copy()
        x, y, new_options = _arg2d(arg)
        options.update(new_options)

        if c is not False:
            options['color'] = colors[i]

        if normalize_x:
            x = _normalise(x)
        if normalize_y:
            y = _normalise(y)

        ax.plot(x, y, **options)

    if c is not False:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=c_label)
        


    return _end(fig, ax, kwargs2, path, show)
    

        
def dots(*args, additional_lines=[], **kwargs):
    fig = kwargs.pop('fig', None)
    ax = kwargs.pop('ax', None)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))  # create new if none provided

    global_options, kwargs2 = _kwargs(ax, kwargs)

    path, show = kwargs.pop('path', None), kwargs.pop('show', True)
    normalize_x, normalize_y = kwargs.pop('normalize_x', False), kwargs.pop('normalize_y', False)
    c = kwargs.get('c', False)
    
    if c is not False:
        if kwargs.pop('c_scale', 'linear') == 'log':
            norm = mcolors.LogNorm(vmin=min(c), vmax=max(c))
        else:
            norm = mcolors.Normalize(vmin=min(c), vmax=max(c))
        cmap = cm.get_cmap(kwargs.get('cmap', 'viridis'))
        c_label = kwargs.pop('c_label', None)
    
    for i, arg in enumerate(args):
        options = global_options.copy()
        x, y, new_options = _arg2d(arg)
        options.update(new_options)

        if normalize_x:
            x = _normalise(x)
        if normalize_y:
            y = _normalise(y)

        ax.scatter(x, y, **options)
        
    _additionalLines(ax, additional_lines, global_options)
    
    if c is not False:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=c_label)
        
    return _end(fig, ax, kwargs2, path, show)

def candles(candles, *args, x_offset=0, alpha=1.0, open_color='green', close_color='red', open_legend='', close_legend='', body_width=0.6, wick_linewidth=1.0, **kwargs):
    fig = kwargs.pop('fig', None)
    ax = kwargs.pop('ax', None)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))  # create new if none provided
        
    global_options, kwargs2 = _kwargs(ax, kwargs)

    path, show = kwargs.pop('path', None), kwargs.pop('show', True)

    candles = np.asarray(candles)
    n = candles.shape[0]
    
    # Determine x positions.
    if np.isscalar(x_offset):
        xs = np.arange(n) + x_offset
    else:
        xs = np.asarray(x_offset)
        if xs.shape[0] != n:
            raise ValueError("x_offset must be a scalar or list-like of length equal to number of candles.")

    # Determine alpha values.
    if np.isscalar(alpha):
        alphas = [alpha] * n
    else:
        alphas = np.asarray(alpha)
        if alphas.shape[0] != n:
            raise ValueError("alpha must be a scalar or list-like of length equal to number of candles.")
    
    for i, candle in enumerate(candles):
        x = xs[i]
        open_, high, low, close = candle
        if close >= open_:
            color = open_color
            body_low = open_
            body_high = close
        else:
            color = close_color
            body_low = close
            body_high = open_
        curr_alpha = alphas[i]

        # Plot the wick (vertical line from low to high)
        ax.plot([x, x], [low, high], color=color, linewidth=wick_linewidth, alpha=curr_alpha)
        # Plot the body (rectangle between open and close)
        rect = plt.Rectangle((x - body_width / 2, body_low), body_width, body_high - body_low,
                             facecolor=color, edgecolor=color, alpha=curr_alpha)
        ax.add_patch(rect)
    
    existing_legend = ax.get_legend()
    prev_handles, prev_labels = [], []
    if existing_legend is not None:
        # pull its handles + labels
        prev_handles = existing_legend.legend_handles
        prev_labels  = [t.get_text() for t in existing_legend.get_texts()]
        
    # build your two new patches
    new_handles = []
    if open_legend:
        new_handles.append(Patch(facecolor=open_color,  edgecolor=open_color,  label=open_legend))
    if close_legend:
        new_handles.append(Patch(facecolor=close_color, edgecolor=close_color, label=close_legend))
    
    if len(new_handles) > 0:
        ax.legend(handles=prev_handles + new_handles, 
                labels=prev_labels  + [open_legend, close_legend], 
                loc='upper right')
  
    return _end(fig, ax, kwargs2, path, show)
        

def heatmap(*args, additional_lines=[], **kwargs):
    fig = kwargs.pop('fig', None)
    ax = kwargs.pop('ax', None)
    
    bins = kwargs.pop('bins', 100)
    cmap_label = kwargs.pop('cmap_label', '')

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))  # create new if none provided

    global_options, kwargs2 = _kwargs(ax, kwargs)

    path, show = kwargs.pop('path', None), kwargs.pop('show', True)
    normalize_x, normalize_y = kwargs.pop('normalize_x', False), kwargs.pop('normalize_y', False)
    
    for i, arg in enumerate(args):
        options = global_options.copy()
        x, y, new_options = _arg2d(arg)
        options.update(new_options)

        if normalize_x:
            x = _normalise(x)
        if normalize_y:
            y = _normalise(y)

        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(heatmap.T, extent=extent, origin='lower', aspect=(xedges[1]-xedges[0]) / (yedges[1]-yedges[0]), **options)
        plt.colorbar(label=cmap_label)
        
    _additionalLines(ax, additional_lines, global_options)
        
    return _end(fig, ax, kwargs2, path, show)
# def dotsHeat(x, y, title='Desire Vs real', x_label='Desire', y_label='Real', 
#             xlim=None, ylim=None, xscale='linear', yscale='linear', 
#             color=None, cmap='Blues', gridsize=50, bins=None,  mincnt=1, additional_lines: list[dict]=[], show=True, path=None): # 
#     hb = plt.hexbin(x, y, cmap=cmap,  mincnt=mincnt, gridsize=gridsize, bins=bins)
#     plt.colorbar(hb)
#     for al in additional_lines:
#         plt.plot(al['x'], al['y'], color=al['color'] if 'color' in al else color, linestyle=al['linestyle'] if 'linestyle' in al else '--')
#     plt.title(title)
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.xscale(xscale)
#     plt.yscale(yscale)

#     if xlim is not None:
#         plt.xlim(xlim)

#     if ylim is not None:
#         plt.ylim(ylim)    

        
#     return _end(plt, path, show)
    
# def graphs(*args, title=None, x_label='', y_label='', color='b', linestyle='-', linewidth=2, grid=True, show=True, path=None):
#     grid = math.sqrt(len(args))
#     nrows, ncols = int(grid), math.ceil(grid)
#     fig = plt.figure()
#     if title:
#         fig.suptitle(title)
    
#     for i, arg in enumerate(args):
#         options = {'title':None, 'color':color, 'linestyle':linestyle, 'linewidth':linewidth, 'grid':grid, 'label': None}
#         x, y = arg[0], arg[1]
#         if len(arg) == 3 and isinstance(arg[2], dict):
#             options.update(arg[2])
        
#         sub = fig.add_subplot(nrows, ncols, i+1)
#         sub.plot(x, y, color=options['color'], linestyle=options['linestyle'], linewidth=options['linewidth'], label=options['label'])
        
#         if options['title']:
#             sub.set_title(options['title'])
            
#         sub.grid(options['grid'])
        
#         if options['label']:
#             fig.legend()
            
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
            
#     fig.tight_layout()
    
#     return _end(plt, path, show)
    

    
# def heatmap(x, y, title='Heatmap', x_label='X', y_label='Y', bins=50, cmap='Blues', show=True, path=None):
#     print(x.shape, y.shape)
#     hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
#     sns.heatmap(hist, xticklabels=xedges, yticklabels=yedges, cmap='Blues')
#     plt.show()
#     # sns.heatmap(matrix)
#     # plt.show()