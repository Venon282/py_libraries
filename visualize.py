import matplotlib.pyplot as plt
import math
from pathlib import Path
import numpy as np
import seaborn as sns

def handleKargs(plt, kwargs):
    
    if kwargs.get('legend', False):
        plt.legend()
        
    if kwargs.get('show', False):
        plt.show()
        
    path = kwargs.get('path', False)
    if path:
        save(plt, file_path=path)

def save(plt, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch(exist_ok=True)
    plt.savefig(file_path)

def curve(x, y, title=None, color='b', linestyle='-', linewidth=2, grid=True):
    """
    Plots a simple curve with customization options.
    
    Parameters:
    - x: The x-values (list or array-like).
    - y: The y-values (list or array-like).
    - title: The title of the plot (optional).
    - xlabel: Label for the x-axis (optional).
    - ylabel: Label for the y-axis (optional).
    - color: Color of the line (default is 'blue').
    - linestyle: Line style (default is solid line '-').
    - linewidth: Width of the line (default is 2).
    - grid: Whether to show the grid (default is True).
    """
    plt.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth)
    
    if title:
        plt.title(title)
        
    plt.grid(grid)
    
    plt.show()
    plt.close()

def curves(*args, title=None, x_label='', y_label='', color=None, linestyle='-', 
           xlim=None, ylim=None, linewidth=2, grid=True, legend=True, show=True, path=None,
           normalize_y=False, **kwargs):
    """
    Plots one or more curves.

    Parameters:
    -----------
    *args : tuple ornot
        Each argument can be:
          - (x, y): x and y data arrays.
          - (y, options): y data and a dict of options (x is generated as range(len(y))).
          - (x, y, options): x and y data arrays with a dict of options.
    title : str, optional
        Title of the plot.
    x_label : str, optional
        Label for the x-axis.
    y_label : str, optional
        Label for the y-axis.
    color : any, optional
        Default color for the curves.
    linestyle : str, optional
        Default linestyle for the curves.
    xlim : tuple, optional
        Limits for the x-axis (e.g., (xmin, xmax)).
    ylim : tuple, optional
        Limits for the y-axis (e.g., (ymin, ymax)). Ignored if normalize_y is True and not provided.
    linewidth : float, optional
        Width of the plotted lines.
    grid : bool, optional
        If True, display the grid.
    legend : bool, optional
        If True, display a legend.
    show : bool, optional
        If True, display the plot.
    path : str, optional
        File path to save the plot. If provided, the plot is saved using plt.savefig().
    normalize_y : bool, optional
        If True, each curve's y data is normalized to the range [0, 1] so that
        all curves touch both the top and bottom of the graph.
    **kwargs :
        Additional keyword arguments (currently unused).

    Returns:
    --------
    None
    """
    
    if title:
        plt.title(title)
    
    xs, ys = [], []
    
    for arg in args:
        # Set default plot options
        options = {'color': color, 'linestyle': linestyle, 'linewidth': linewidth, 'label': None}
        # Support for (x, y, options) or (y, options) input formats
        if isinstance(arg, tuple):
            if len(arg) == 3 and isinstance(arg[2], dict):
                options.update(arg[2])
                x, y = arg[0], arg[1]
            elif len(arg) == 2:
                if isinstance(arg[1], dict):
                    options.update(arg[1])
                    y = arg[0]
                    x = np.arange(len(y))
                else:
                    x, y = arg[0], arg[1]
            else:
                x , y = np.arange(len(arg[0])), arg[0]
        else:
            x, y = np.arange(len(arg)), arg
        
        # Normalize y values to [0, 1] if required
        if normalize_y:
            y = np.array(y)
            y_min = np.min(y)
            y_max = np.max(y)
            if y_max != y_min:
                y = (y - y_min) / (y_max - y_min)
            else:
                # For a constant curve, use a middle value
                y = np.full_like(y, 0.5)
        
        xs.append(x)
        ys.append(y)
        
        plt.plot(x, y, color=options['color'], linestyle=options['linestyle'],
                 linewidth=options['linewidth'], label=options['label'])
    
    # Apply grid and axis labels
    if grid:
        plt.grid(True)
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Set x-axis and y-axis limits
    if xlim is not None:
        plt.xlim(xlim)
    
    # If normalization is enabled and no ylim is provided, force y-axis to [0, 1]
    if normalize_y and ylim is None:
        plt.ylim(0, 1)
    elif ylim is not None:
        plt.ylim(ylim)
            
    if legend:
        plt.legend()
        
    # Save the plot if a path is provided
    if path is not None:
        save(plt, file_path=path) # plt.savefig(path)
    
    if show:
        plt.show()
    plt.close()

        
def dots(x, y, s=None, title='Desire Vs real', x_label='Desire', y_label='Real', 
            xlim=None, ylim=None, xscale='linear', yscale='linear',
            color='red', additional_lines: list[dict]=[], show=True, path=None):
    plt.scatter(x, y, s, alpha=0.4, color=color)
    for al in additional_lines:
        plt.plot(al['x'], al['y'], color=al['color'] if 'color' in al else color, linestyle=al['linestyle'] if 'linestyle' in al else '--')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xscale(xscale)
    plt.yscale(yscale)

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)    
    if show:
        plt.show()
        
    if path is not None:
        save(plt, file_path=path)
    plt.close()
      
def dotsHeat(x, y, title='Desire Vs real', x_label='Desire', y_label='Real', 
            xlim=None, ylim=None, xscale='linear', yscale='linear', 
            color=None, cmap='Blues', gridsize=50, bins=None,  mincnt=1, additional_lines: list[dict]=[], show=True, path=None): # 
    hb = plt.hexbin(x, y, cmap=cmap,  mincnt=mincnt, gridsize=gridsize, bins=bins)
    plt.colorbar(hb)
    for al in additional_lines:
        plt.plot(al['x'], al['y'], color=al['color'] if 'color' in al else color, linestyle=al['linestyle'] if 'linestyle' in al else '--')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xscale(xscale)
    plt.yscale(yscale)

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)    

        
    if show:
        plt.show()
        
    if path is not None:
        save(plt, file_path=path)
    plt.close()
    
def graphs(*args, title=None, x_label='', y_label='', color='b', linestyle='-', linewidth=2, grid=True, show=True, path=None):
    grid = math.sqrt(len(args))
    nrows, ncols = int(grid), math.ceil(grid)
    fig = plt.figure()
    if title:
        fig.suptitle(title)
    
    for i, arg in enumerate(args):
        options = {'title':None, 'color':color, 'linestyle':linestyle, 'linewidth':linewidth, 'grid':grid, 'label': None}
        x, y = arg[0], arg[1]
        if len(arg) == 3 and isinstance(arg[2], dict):
            options.update(arg[2])
        
        sub = fig.add_subplot(nrows, ncols, i+1)
        sub.plot(x, y, color=options['color'], linestyle=options['linestyle'], linewidth=options['linewidth'], label=options['label'])
        
        if options['title']:
            sub.set_title(options['title'])
            
        sub.grid(options['grid'])
        
        if options['label']:
            fig.legend()
            
    plt.xlabel(x_label)
    plt.ylabel(y_label)
            
    fig.tight_layout()
    if show:
        plt.show()
        
    if path is not None:
        save(plt, file_path=path)
    plt.close()
    
# def heatmap(x, y, title='Heatmap', x_label='X', y_label='Y', bins=50, cmap='Blues', show=True, path=None):
#     print(x.shape, y.shape)
#     hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
#     sns.heatmap(hist, xticklabels=xedges, yticklabels=yedges, cmap='Blues')
#     plt.show()
#     # sns.heatmap(matrix)
#     # plt.show()