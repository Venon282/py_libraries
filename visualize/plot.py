import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from typing import Any, Dict, List, Optional, Tuple, Union

# internal
from .main import save, makeBins

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
        path = kwargs.pop('path', kwargs.pop('save', None))
        show = kwargs.pop('show', True)

        # styling
        style = {
            'title':  kwargs.pop('title', ''),
            'xlabel': kwargs.pop('xlabel', ''),
            'ylabel': kwargs.pop('ylabel', ''),
            'y2label': kwargs.pop('y2label', ''),
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
        if style['y2label'] != '':
            #todo modify the class for handle more than 1 ax
            ax2 = ax.twinx()
            ax2.set_ylabel(style['y2label'])
            
        grid = style['grid']
        
        ax.minorticks_on()
        if isinstance(grid, bool):
            ax.grid(grid)
        elif isinstance(grid, dict):
            ax.grid(True, **grid)
        else: raise TypeError(f'Unsupported grid type {type(grid)}. Should be either a bool or a dict.')
        
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
            elif isinstance(y, dict):
                opts = y
                y = x
                x = np.arange(len(y))
            x = np.asarray(x); y = np.asarray(y)
        else:
            raise ValueError(f"Can't parse argument {arg}")
        return x, y, opts
    
    @staticmethod
    def _end(fig, ax, style, extras, path, show):
        Plot._apply_style(fig, ax, style, extras)
        if path:    save(fig, path)
        if show:    plt.show(); return None
        if extras['close']:   plt.close(fig)
        return fig, ax

    @staticmethod
    def plot(*args, cmap=None, cmap_label='', cmap_values=None, **kwargs) -> Optional[Figure]:
        """Simple line plot."""
        fig, ax, style, extras, path, show, gb_opts = Plot._init(kwargs)
        
        if cmap:
            cmap_obj = cm.get_cmap(cmap)
            if cmap_values is not None:
                # Normalize cmap_values to [0,1]
                normed = (np.array(cmap_values) - np.min(cmap_values)) / (np.max(cmap_values) - np.min(cmap_values))
                colors = cmap_obj(normed)
            else:
                # Evenly spaced colors for each line
                colors = cmap_obj(np.linspace(0, 1, len(args)))
            
        for i, arg in enumerate(args):
            x, y, opts = Plot._unwrap(arg)
            
            if cmap and colors[i] is not None:
                opts['color'] = colors[i]
                
            ax.plot(x, y, **{**gb_opts, **opts})
        
        if cmap:
            to_norm = colors if cmap_values is None else cmap_values
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=to_norm.min(), vmax=to_norm.max()))
            fig.colorbar(sm, ax=ax, label=cmap_label)
        
        return Plot._end(fig, ax, style, extras, path, show)


    @staticmethod
    def scatter(*args, **kwargs) -> Optional[Figure]:
        """Scatter / dot plot."""
        fig, ax, style, extras, path, show, gb_opts = Plot._init(kwargs)
        for arg in args:
            x, y, opts = Plot._unwrap(arg)
            ax.scatter(x, y, **{**gb_opts, **opts})
        return Plot._end(fig, ax, style, extras, path, show)
    
    @staticmethod
    def image(*args, **kwargs) -> Optional[Figure]:
        """Display an image or a set of images."""
        fig, ax, style, extras, path, show, gb_opts = Plot._init(kwargs)
        colorbar = gb_opts.pop('colorbar',False)
        
        for arg in args:
            _, img, opts = Plot._unwrap(arg)
            im = ax.imshow(img, **{**gb_opts, **opts})
        
        if colorbar:
            fig.colorbar(im, ax=ax, label=kwargs.get('cbar_label',''))
        # Remove axes if not explicitly kept
        # if not kwargs.get("show_axes", False):
        #     ax.axis("off")
        
        return Plot._end(fig, ax, style, extras, path, show)
    
    @staticmethod
    def errorbar(*args, **kwargs) -> Optional[Figure]:
        """Scatter / dot plot."""
        fig, ax, style, extras, path, show, gb_opts = Plot._init(kwargs)
        for arg in args:
            x, y, opts = Plot._unwrap(arg)
            ax.errorbar(np.mean(x), np.mean(y), xerr=np.std(x), yerr=np.std(y), **{**gb_opts, **opts})
        return Plot._end(fig, ax, style, extras, path, show)
    
    
    @staticmethod
    def scatter_density(*args, alpha=False, center=False, rate=False, bw_method: Optional[Union[str, float]] = None, cbar_label: Optional[str] = None, **kwargs) -> Optional[Figure]:
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
        - center: If true, center (the last plot, useless if many args) on the 90% denser values. If a number on it's pourcentage or rate
        """
        import scipy.stats as scipy_stats
        fig, ax, style, extras, path, show, gb_opts = Plot._init(kwargs)
        
        legend_handles = []
        for arg in args:
            x, y, opts = Plot._unwrap(arg)

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
            
            if center:
                rate = 0.9 if center is True else center if 0.0 < center <= 1.0 else float(center) / 100
                sub_idx = int((1.0-rate) * len(x))
                sub_x, sub_y = x[sub_idx:], y[sub_idx:]
                x_min, x_max = min(sub_x), max(sub_x)
                y_min, y_max = min(sub_y), max(sub_y)
                x_center, y_center = sub_x[-1], sub_y[-1]
                x_diffs, y_diffs = np.abs([x_center - x_min, x_max - x_center]), np.abs([y_center - y_min, y_max - y_center])
                x_max_diff, y_max_diff = max(x_diffs), max(y_diffs)
                extras['xlim'] = [x_center - x_max_diff, x_center + x_max_diff]
                extras['ylim'] = [y_center - y_max_diff, y_center + y_max_diff]
                
            
                
            sc = ax.scatter(x, y,c=z, alpha=z_norm,**opts)
            if opts.get('label', ''):
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

        return Plot._end(fig, ax, style, extras, path, show)
    
    @staticmethod
    def histogram(data: Union[List, np.ndarray], bins: int=30, bin_type='linear', **kwargs) -> Optional[Figure]:
        """Univariate histogram."""
        fig, ax, style, extras, path, show, gb_opts = Plot._init(kwargs)
        if bin_type!='linear':
            bins = makeBins(data, bins, bin_type=bin_type)
        ax.hist(data, bins=bins, **gb_opts)
        return Plot._end(fig, ax, style, extras, path, show)

    @staticmethod
    def heatmap(*args, bins: int=100, cmap: str='viridis', **kwargs) -> Optional[Figure]:
        """2D density heatmap."""
        fig, ax, style, extras, path, show, gb_opts = Plot._init(kwargs)
        for arg in args:
            x, y, opts = Plot._unwrap(arg)
            heat, xedges, yedges = np.histogram2d(x, y, bins=bins)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax.imshow(
                heat.T, extent=extent, origin='lower',
                aspect='auto', cmap=cm.get_cmap(cmap),
                **gb_opts, **opts
            )
            fig.colorbar(im, ax=ax, label=kwargs.get('cbar_label',''))
        return Plot._end(fig, ax, style, extras, path, show)

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
        fig, ax, style, extras, path, show, gb_opts = Plot._init(kwargs)
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
        return Plot._end(fig, ax, style, extras, path, show)