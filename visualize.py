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
import matplotlib.markers as mmarkers
from matplotlib.lines import Line2D

import colorsys
import random
from matplotlib import cm, colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

class Visualize:

    class Maker:
        @staticmethod
        def all():
            """All named markers."""
            return list(mmarkers.MarkerStyle.markers.keys())
        
        @staticmethod
        def all_not_none():
            """All named markers."""
            return list(set(mmarkers.MarkerStyle.markers.keys()) - set(Visualize.Maker.none()))

        # — filled vs. unfilled —————

        @staticmethod
        def filled():
            return list(mmarkers.MarkerStyle.filled_markers)

        @staticmethod
        def unfilled():
            return [
                m for m in Visualize.Maker.all()
                if m not in Visualize.Maker.filled()
            ]

        @staticmethod
        def fillstyles():
            return list(mmarkers.MarkerStyle.fillstyles)

        # — geometric shapes —————

        @staticmethod
        def pixel():
            return ['.', ',']

        @staticmethod
        def circle():
            return ['o']

        @staticmethod
        def square():
            return ['s']

        @staticmethod
        def diamond():
            return ['D', 'd']

        @staticmethod
        def triangle():
            return ['v', '^', '<', '>', '1', '2', '3', '4']

        @staticmethod
        def pentagon():
            return ['p', 'P']

        @staticmethod
        def hexagon():
            return ['h', 'H']

        @staticmethod
        def star():
            return ['*', 'X']

        @staticmethod
        def cross():
            return ['+', 'x']

        # — numeric / digit markers —————

        @staticmethod
        def numeric():
            return [
                m for m in Visualize.Maker.all()
                if isinstance(m, int) or (isinstance(m, str) and m.isdigit())
            ]

        # — text‐style / functional —————

        @staticmethod
        def tick():
            return ['|', '_']

        @staticmethod
        def none():
            return ['None', 'none', ' ', '']

        # —组合 for directional —————

        @staticmethod
        def directional():
            return Visualize.Maker.tick() + Visualize.Maker.caret()

        # — everything else —————

        @staticmethod
        def other():
            known = (
                Visualize.Maker.pixel()   + Visualize.Maker.circle() +
                Visualize.Maker.square()  + Visualize.Maker.diamond() +
                Visualize.Maker.triangle()+ Visualize.Maker.pentagon() +
                Visualize.Maker.hexagon() + Visualize.Maker.star()    +
                Visualize.Maker.cross()   + Visualize.Maker.numeric() +
                Visualize.Maker.tick()  +
                Visualize.Maker.none()
            )
            return [m for m in Visualize.Maker.all() if m not in known]

        # — sanity check —————

        @staticmethod
        def verify():
            recognized = (
                Visualize.Maker.pixel()   + Visualize.Maker.circle() +
                Visualize.Maker.square()  + Visualize.Maker.diamond() +
                Visualize.Maker.triangle()+ Visualize.Maker.pentagon() +
                Visualize.Maker.hexagon() + Visualize.Maker.star()    +
                Visualize.Maker.cross()   + Visualize.Maker.numeric() +
                Visualize.Maker.tick()    +
                Visualize.Maker.none()
            )
            return [m for m in recognized if m not in Visualize.Maker.all()]

    class Cmap:
        @staticmethod
        def all():
            """All named colorscales (with both normal and reversed names)."""
            return list(plt.colormaps())

        # Sequential (single-hue or multi-hue)
        @staticmethod
        def sequential():
            return [
                'magma', 'inferno', 'plasma', 'viridis', 'cividis',
                'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'binary', 'BuGn', 'BuPu', 'GnBu', 'OrRd',
                'PuBu', 'PuBuGn', 'PuRd', 'RdPu', 'YlGn', 'YlGnBu',
                # additional sequential/miscellaneous
                # 'afmhot', 'autumn', 'bone', 'brg', 'cool', 'copper',
                # 'cubehelix', 'flag', 'gist_earth', 'gist_gray', 'gist_heat', 'gist_ncar',
                # 'gnuplot', 'gnuplot2', 'gray', 'hot', 'jet', 'nipy_spectral',
                # 'ocean', 'pink', 'prism', 'rainbow', 'summer', 'terrain', 'winter',
                # 'turbo', 'spring', 'seismic', 'gist_rainbow', 'gist_stern', 'gist_yarg',
                # 'grey', 'gist_grey', 'gist_yerg', 'Grays', 'rocket'
            ]

        @staticmethod
        def sequential_r():
            all_ = Visualize.Cmap.all()
            return [m + '_r' for m in Visualize.Cmap.sequential() if m + '_r' in all_]

        # Diverging
        @staticmethod
        def diverging():
            return [
                'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy',
                'RdYlBu', 'RdBu', 'RdYlGn', 'Spectral',
                'coolwarm', 'bwr'
            ]

        @staticmethod
        def diverging_r():
            all_ = Visualize.Cmap.all()
            return [m + '_r' for m in Visualize.Cmap.diverging() if m + '_r' in all_]

        # Cyclical
        @staticmethod
        def cyclical():
            return [
                'twilight', 'twilight_shifted', 'hsv',
                # additional cyclic
                'flag', 'prism'
            ]

        @staticmethod
        def cyclical_r():
            all_ = Visualize.Cmap.all()
            return [m + '_r' for m in Visualize.Cmap.cyclical() if m + '_r' in all_]

        # Qualitative (discrete/category)
        @staticmethod
        def qualitative():
            return [
                'tab10', 'tab20', 'tab20b', 'tab20c',
                'Pastel1', 'Pastel2', 'Paired', 'Accent',
                'Dark2', 'Set1', 'Set2', 'Set3',
                # others
                'Wistia', 'CMRmap', 'flare', 'crest', 'icefire', 'mako', 'vlag'
            ]

        @staticmethod
        def qualitative_r():
            all_ = Visualize.Cmap.all()
            return [m + '_r' for m in Visualize.Cmap.qualitative() if m + '_r' in all_]

        @staticmethod
        def other():
            all_ = Visualize.Cmap.all()
            known = (
                Visualize.Cmap.sequential() + Visualize.Cmap.sequential_r() +
                Visualize.Cmap.diverging() + Visualize.Cmap.diverging_r() +
                Visualize.Cmap.cyclical() + Visualize.Cmap.cyclical_r() +
                Visualize.Cmap.qualitative() + Visualize.Cmap.qualitative_r()
            )
            return [a for a in all_ if a not in known]

        @staticmethod
        def verify():
            all_ = Visualize.Cmap.all()
            recognized = (
                Visualize.Cmap.sequential() + Visualize.Cmap.sequential_r() +
                Visualize.Cmap.diverging() + Visualize.Cmap.diverging_r() +
                Visualize.Cmap.cyclical() + Visualize.Cmap.cyclical_r() +
                Visualize.Cmap.qualitative() + Visualize.Cmap.qualitative_r()
            )
            return [r for r in recognized if r not in all_]

    class Color:
        @staticmethod
        def all():
            """All named colors from matplotlib (base, CSS4, Tableau, XKCD)."""
            return list(mcolors.get_named_colors_mapping().keys())

        @staticmethod
        def base():
            """Base colors (short names)."""
            return list(mcolors.BASE_COLORS.keys())

        @staticmethod
        def tableau():
            """Tableau colors (named palette colors)."""
            return list(mcolors.TABLEAU_COLORS.keys())

        @staticmethod
        def css4():
            """CSS4 named colors."""
            return list(mcolors.CSS4_COLORS.keys())

        @staticmethod
        def xkcd():
            """XKCD survey colors (prefixed names)."""
            return list(mcolors.XKCD_COLORS.keys())

        @staticmethod
        def hex_values():
            """All hex values of named colors."""
            return list(mcolors.get_named_colors_mapping().values())

        @staticmethod
        def other():
            """Colors not in base, tableau, CSS4, or XKCD."""
            all_ = set(Visualize.Color.all())
            known = set(Visualize.Color.base() + Visualize.Color.tableau() + Visualize.Color.css4() + Visualize.Color.xkcd())
            return list(all_ - known)

        @staticmethod
        def luminance(color_name):
            """Return relative luminance [0..1] of a named color."""
            rgb = mcolors.to_rgb(mcolors.get_named_colors_mapping()[color_name])
            # Standard Rec. 709 luminance
            return 0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]

        @staticmethod
        def light():
            """Named colors whose names start with 'light' (case-insensitive)."""
            return [name for name in Visualize.Color.all() if name.lower().startswith('light')]

        @staticmethod
        def dark():
            """Named colors whose names start with 'dark' (case-insensitive)."""
            return [name for name in Visualize.Color.all() if name.lower().startswith('dark')]

        @staticmethod
        def dark(threshold=0.5):
            """Named colors with luminance at or below threshold (dark colors)."""
            return [name for name in Visualize.Color.all() if Visualize.Color.luminance(name) <= threshold]

        @staticmethod
        def grayscale(tolerance=1e-6):
            """Named colors where R≈G≈B within tolerance."""
            result = []
            for name, hexcode in mcolors.get_named_colors_mapping().items():
                r, g, b = mcolors.to_rgb(hexcode)
                if abs(r-g) < tolerance and abs(g-b) < tolerance:
                    result.append(name)
            return result

        @staticmethod
        def primary(dominance=0.1):
            """Colors where one channel exceeds the other two by 'dominance'."""
            prim = {'red': [], 'green': [], 'blue': []}
            for name, hexcode in mcolors.get_named_colors_mapping().items():
                r, g, b = mcolors.to_rgb(hexcode)
                if r - max(g, b) > dominance:
                    prim['red'].append(name)
                if g - max(r, b) > dominance:
                    prim['green'].append(name)
                if b - max(r, g) > dominance:
                    prim['blue'].append(name)
            return prim

        @staticmethod
        def complementary(color_name):
            """Return hex code for the complementary color of a named color."""
            hexcode = mcolors.get_named_colors_mapping()[color_name]
            r, g, b = mcolors.to_rgb(hexcode)
            comp = (1-r, 1-g, 1-b)
            return mcolors.to_hex(comp)

        @staticmethod
        def lighten(color_name, factor=0.2):
            """Lighten a named color by a given factor (0..1)."""
            hexcode = mcolors.get_named_colors_mapping()[color_name]
            r, g, b = mcolors.to_rgb(hexcode)
            r, g, b = [min(1, c + factor*(1-c)) for c in (r, g, b)]
            return mcolors.to_hex((r, g, b))

        @staticmethod
        def darken(color_name, factor=0.2):
            """Darken a named color by a given factor (0..1)."""
            hexcode = mcolors.get_named_colors_mapping()[color_name]
            r, g, b = mcolors.to_rgb(hexcode)
            r, g, b = [max(0, c*(1-factor)) for c in (r, g, b)]
            return mcolors.to_hex((r, g, b))

        @staticmethod
        def sorted_by_hue(include_hex=False):
            """Return colors sorted by HSV hue. Optionally include hex codes."""
            items = []
            for name, hexcode in mcolors.get_named_colors_mapping().items():
                r, g, b = mcolors.to_rgb(hexcode)
                h, l, s = colorsys.rgb_to_hls(r, g, b)
                items.append((h, name, hexcode))
            items.sort(key=lambda x: x[0])
            return [(name, hex) if include_hex else name for _, name, hex in items]

        @staticmethod
        def random(n=5, seed=None):
            """Random subset of n named colors."""
            names = Visualize.Color.all()
            if seed is not None:
                random.seed(seed)
            return random.sample(names, min(n, len(names)))

        @staticmethod
        def verify():
            """Verify that named mappings are consistent and unique."""
            mapping = mcolors.get_named_colors_mapping()
            names = list(mapping.keys())
            unique = len(names) == len(set(names))
            duplicates = [name for name in set(names) if names.count(name) > 1]
            return {
                'total': len(names),
                'unique': unique,
                'duplicates': duplicates
            }

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
                'legend_opts':kwargs.pop('legend_opts', {}),
                'close':kwargs.pop('close', False),
            }
            return fig, ax, style, extras, path, show, kwargs

        @staticmethod
        def _apply_style(fig: Figure, ax: Axes,
                         style: Dict[str, Any], extras: Dict[str, Any]) -> None:
            if style['title'] != '': ax.set_title(style['title'])
            if style['xlabel'] != '': ax.set_xlabel(style['xlabel'])
            if style['ylabel'] != '': ax.set_ylabel(style['ylabel'])
            ax.grid(style['grid'])
            if extras['xlim'] is not None:   ax.set_xlim(extras['xlim'])
            if extras['ylim'] is not None:   ax.set_ylim(extras['ylim'])
            if extras['xscale'] is not None: ax.set_xscale(extras['xscale'])
            if extras['yscale'] is not None: ax.set_yscale(extras['yscale'])
            if extras['legend']: ax.legend(**extras['legend_opts'])
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
            if extras['close']:   plt.close(fig)
            return fig, ax

        @staticmethod
        def plot(*args, **kwargs) -> Optional[Figure]:
            """Simple line plot."""
            fig, ax, style, extras, path, show, gb_opts = Visualize.Plot._init(kwargs)
            for arg in args:
                x, y, opts = Visualize.Plot._unwrap(arg)
                
                ax.plot(x, y, **{**gb_opts, **opts})
            return Visualize.Plot._end(fig, ax, style, extras, path, show)


        @staticmethod
        def scatter(*args, **kwargs) -> Optional[Figure]:
            """Scatter / dot plot."""
            fig, ax, style, extras, path, show, gb_opts = Visualize.Plot._init(kwargs)
            for arg in args:
                x, y, opts = Visualize.Plot._unwrap(arg)
                ax.scatter(x, y, **{**gb_opts, **opts})
            return Visualize.Plot._end(fig, ax, style, extras, path, show)
        
        @staticmethod
        def errorbar(*args, **kwargs) -> Optional[Figure]:
            """Scatter / dot plot."""
            fig, ax, style, extras, path, show, gb_opts = Visualize.Plot._init(kwargs)
            for arg in args:
                x, y, opts = Visualize.Plot._unwrap(arg)
                ax.errorbar(np.mean(x), np.mean(y), xerr=np.std(x), yerr=np.std(y), **{**gb_opts, **opts})
            return Visualize.Plot._end(fig, ax, style, extras, path, show)
        
        
        @staticmethod
        def scatter_density(*args, alpha=False, rate=False, bw_method: Optional[Union[str, float]] = None, cbar_label: Optional[str] = None, **kwargs) -> Optional[Figure]:
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
            fig, ax, style, extras, path, show, gb_opts = Visualize.Plot._init(kwargs)
            
            legend_handles = []
            for arg in args:
                x, y, opts = Visualize.Plot._unwrap(arg)

                # compute 2D KDE
                xy = np.vstack([x, y])
                kde = scipy_stats.gaussian_kde(xy, bw_method=bw_method)
                z = kde(xy)
                idx = z.argsort() # sort the points by density, so that high-density points are on top (of the plot, bottom of the list)
                
                if rate:
                    if isinstance(rate, (float)):
                        idx = idx[int(len(idx) * (1-rate)):]
                    elif isinstance(rate, int):
                        idx = idx[max(0, len(idx) - rate):]
                    else:
                        print(f'Warning: Rate of type {type(rate)} with the value {rate} can not take into consideration. Waiting a float or int.')
                     
                x, y, z = x[idx], y[idx], z[idx] # Apply the potential rate and order
                # Allow do add a visibility lebel
                if alpha:
                    # z is sorted so we know where is the max and min
                    z_norm = (z - z[0]) / (z[-1] - z[0])
                else: z_norm = None
                
                    
                sc = ax.scatter(x, y,c=z, alpha=z_norm,**opts)
                if opts.get('label', None):
                    legend_handles.append(Line2D(
                        [0], [0],
                        marker=opts.get('marker', 'o'), # Match marker shape
                        color='none',                 # No line
                        markeredgecolor=opts.get('edgecolor', 'none'),
                        markerfacecolor=sc.get_cmap()(0.5),   # Midpoint color
                        markersize=sc.get_sizes()[0]**0.5,  # Convert area to radius
                        label=opts['label']
                    ))

            # add a colorbar for density
            if cbar_label is not None:
                fig.colorbar(sc, ax=ax, label=cbar_label)
                
            if extras['legend']:
                extras['legend_opts'].update({'handles':legend_handles})

            return Visualize.Plot._end(fig, ax, style, extras, path, show)
        
        @staticmethod
        def histogram(data: Union[List, np.ndarray], bins: int=30, **kwargs) -> Optional[Figure]:
            """Univariate histogram."""
            fig, ax, style, extras, path, show, gb_opts = Visualize.Plot._init(kwargs)
            ax.hist(data, bins=bins, **gb_opts)
            return Visualize.Plot._end(fig, ax, style, extras, path, show)

        @staticmethod
        def heatmap(*args, bins: int=100, cmap: str='viridis', **kwargs) -> Optional[Figure]:
            """2D density heatmap."""
            fig, ax, style, extras, path, show, gb_opts = Visualize.Plot._init(kwargs)
            for arg in args:
                x, y, opts = Visualize.Plot._unwrap(arg)
                heat, xedges, yedges = np.histogram2d(x, y, bins=bins)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                im = ax.imshow(
                    heat.T, extent=extent, origin='lower',
                    aspect='auto', cmap=cm.get_cmap(cmap),
                    **{**gb_opts, **opts}
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
            fig, ax, style, extras, path, show, gb_opts = Visualize.Plot._init(kwargs)
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