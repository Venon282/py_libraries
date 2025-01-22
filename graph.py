import matplotlib.pyplot as plt
import math
from pathlib import Path

def save(plt, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch(exist_ok=True)
    plt.savefig(file_path)
    plt.close()

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
    
def curves(*args, title=None, x_label='', y_label='', color=None, linestyle='-', linewidth=2, grid=True, legend=True, show=True, path=None):
    if title:
        plt.title(title)
    
    for i, arg in enumerate(args):
        options = {'color':color, 'linestyle':linestyle, 'linewidth':linewidth, 'label': None}
        x, y = arg[0], arg[1]
        if len(arg) == 3 and isinstance(arg[2], dict):
            options.update(arg[2])
        
        plt.plot(x, y, color=options['color'], linestyle=options['linestyle'], linewidth=options['linewidth'], label=options['label'])
            
    plt.grid(grid)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
            
    if legend:
        plt.legend()
        
    if show:
        plt.show()
        
    if path is not None:
        save(plt, file_path=path)
        
def dots(x_desire, y_real, title='Desire Vs real', x_label='Desire', y_label='Real', 
            xlim=None, ylim=None,
            color='red', additional_lines: list[dict]=[], show=True, path=None):
    plt.scatter(x_desire, y_real, alpha=0.4, color=color)
    for al in additional_lines:
        plt.plot(al['x'], al['y'], color=al['color'] if 'color' in al else color, linestyle=al['linestyle'] if 'linestyle' in al else '--')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)    
    if show:
        plt.show()
        
    if path is not None:
        save(plt, file_path=path)
      
def dotsHeat(x_desire, y_real, title='Desire Vs real', x_label='Desire', y_label='Real', 
            xlim=None, ylim=None,
            color=None, cmap='Blues', gridsize=50, mincnt=1, additional_lines: list[dict]=[], show=True, path=None):
    plt.hexbin(x_desire, y_real, cmap=cmap, mincnt=mincnt, gridsize=gridsize)
    for al in additional_lines:
        plt.plot(al['x'], al['y'], color=al['color'] if 'color' in al else color, linestyle=al['linestyle'] if 'linestyle' in al else '--')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)    
    if show:
        plt.show()
        
    if path is not None:
        save(plt, file_path=path)
    
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
    
