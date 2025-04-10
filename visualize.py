import matplotlib.pyplot as plt
import math
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.lines as mlines

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
    _kwargs2(ax, kwargs2)
    
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

def _kwargs2(ax, kwargs):
    if kwargs.pop('legend', False):
        ax.legend()
    
def _kwargs(ax, kwargs):
    def apply(kwargs, label, function):
        element =  kwargs.pop(label, False)
        if element is not False:
            function(element)
            
    ax.set_title(kwargs.pop('title', ''))
    ax.grid(kwargs.pop('grid', None))
    ax.set_xlabel(kwargs.pop('xlabel', None))
    ax.set_ylabel(kwargs.pop('ylabel', None))
    
    apply(kwargs, 'ylim', ax.set_ylim)
    apply(kwargs, 'xlim', ax.set_xlim)
    
    apply(kwargs, 'xscale', ax.set_xscale)
    apply(kwargs, 'yscale', ax.set_yscale)
        
    kwargs2 = {'legend': kwargs.pop('legend', False)}

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