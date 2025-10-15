import numpy as np
from pathlib import Path


import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from matplotlib import cm, colors


    
            
        
# ------------------------------------------
# GENERAL FUNCTIONS
# ------------------------------------------

# def save(fig, file_path):
#     file_path = Path(file_path)
#     file_path.parent.mkdir(parents=True, exist_ok=True)
#     file_path.touch(exist_ok=True)
#     fig.savefig(str(file_path))
    




    
# def _end(fig, ax, kwargs2, path, show):
#     """
#     Returns:
#         plt: if don't want to save or show
#         None: if show or save
#     """
#     _kwargs2(fig, ax, kwargs2)
    
#     fig.tight_layout()
#     # Save the plot if a path is provided
#     if path is not None:
#         save(fig, file_path=path) # plt.savefig(path)
    
#     if show:
#         plt.show()
#     elif path is None:
#         return fig    
#     else:
#         plt.close(fig)
    
# def _normalise(c):
#     c = np.array(c)
#     c_min = np.min(c)
#     c_max = np.max(c)
#     if c_max != c_min:
#         return (c - c_min) / (c_max - c_min)
#     else:
#         # For a constant curve, use a middle value
#         return np.full_like(c, 0.5)
    
# def _arg2d(arg):
#     options = {}
#     # Support for (x, y, options) or (y, options) input formats
#     if isinstance(arg, (tuple, list)):
#         if len(arg) == 3:
#             if isinstance(arg[2], dict):
#                 options.update(arg[2])
#                 x, y = arg[0], arg[1]
#             else: raise AttributeError('Third argument in args not handle')
#         elif len(arg) == 2:
#             if isinstance(arg[1], dict):
#                 options.update(arg[1])
#                 y = arg[0]
#                 x = np.arange(len(y))
#             else:
#                 x, y = arg[0], arg[1]
#         elif len(arg) == 1:
#             x , y = np.arange(len(arg[0])), arg[0]
#         else: raise AttributeError(f'Dimension {len(arg)} not handle')
#     else:
#         x, y = np.arange(len(arg)), arg

#     return x, y, options

# def _kwargs2(fig, ax, kwargs):
    

#     if kwargs.pop('legend_fig', False) or kwargs.pop('legend', False):
#         fig.legend()
#     elif kwargs.pop('legend_ax', False):
#         ax.legend()

#     _apply(kwargs, 'xscale', ax.set_xscale)
#     _apply(kwargs, 'yscale', ax.set_yscale)
    
#     ax.relim()          # Recalcule les limites des données
#     ax.autoscale_view() # Applique ces limites à la vue
        
# def _apply(kwargs, label, function):
#     element =  kwargs.pop(label, False)
#     if element is not False:
#         function(element)
    
# def _kwargs(ax, kwargs):
    
            
#     ax.set_title(kwargs.pop('title', ''))
#     ax.grid(kwargs.pop('grid', None))
#     ax.set_xlabel(kwargs.pop('xlabel', None))
#     ax.set_ylabel(kwargs.pop('ylabel', None))
    
#     _apply(kwargs, 'ylim', ax.set_ylim)
#     _apply(kwargs, 'xlim', ax.set_xlim)
        
#     kwargs2 = {'legend': kwargs.pop('legend', False), 'legend_ax': kwargs.pop('legend_ax', False), 'legend_fig': kwargs.pop('legend_fig', False), 'xscale': kwargs.pop('xscale', False), 'yscale': kwargs.pop('yscale', False)}

#     return kwargs, kwargs2

# def _additionalLines(ax, additional_lines, global_options):
#     if len(additional_lines) == 0:
#         return
    
#     # Get the list of all valid keyword arguments for Line2D
#     dummy_line = Line2D([], [])  # Create a dummy Line2D object to access its properties
#     valid_props = set(dummy_line.properties().keys())  # returns a set of valid property names

#     for al in additional_lines:
#         options = global_options.copy()
#         x, y, new_options = _arg2d(al)
#         options.update(new_options)
#         # Filter out options that are not valid for Line2D
#         valid_options = {key: value for key, value in options.items() if key in valid_props}
#         ax.plot(x, y, **valid_options)
