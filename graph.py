import matplotlib.pyplot as plt
import math

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
    
def curves(*args, title=None, color='b', linestyle='-', linewidth=2, grid=True):
    grid = math.sqrt(len(args))
    nrows, ncols = int(grid), math.ceil(grid)
    fig = plt.figure()
    if title:
        fig.suptitle(title)
    
    for i, arg in enumerate(args):
        options = {'title':None, 'color':color, 'linestyle':linestyle, 'linewidth':linewidth, 'grid':grid}
        x, y = arg[0], arg[1]
        if len(arg) == 3 and isinstance(arg[2], dict):
            options.update(arg[2])
        
        sub = fig.add_subplot(nrows, ncols, i+1)
        sub.plot(x, y, color=options['color'], linestyle=options['linestyle'], linewidth=options['linewidth'])
        
        if options['title']:
            sub.set_title(options['title'])
            
        sub.grid(options['grid'])
            
    fig.tight_layout()
    plt.show()