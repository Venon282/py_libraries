import numpy as np
from itertools import product
from tqdm import tqdm
import random

def stratified(*args, verbose=0):
    """
    Generates stratified random samples within specified ranges for each parameter.
    
    The input parameters for each dimension should be provided as a tuple with the following structure:
    (bound_min, bound_max (exclusive), n_values, method).
    
    - bound_min: Lower bound of the parameter's range (inclusive).
    - bound_max: Upper bound of the parameter's range (exclusive).
    - n_values: Number of samples to generate per parameter.
    - method: Sampling method (''). Defaults to 'linear'.
    
    Sampling is performed for each dimension separately, with the specified number of bins 
    and the chosen method (linear or logarithmic scaling). A random value is then chosen 
    from each bin to generate a stratified sample.

    Parameters:
    *args (tuples): For each dimension, a tuple of (bound_min, bound_max (exclusive), n_values, method).
                    Method can be 'linear', 'log', 'uniform', 'gaussian', 'exponential', or 'beta'.
    verbose (int): 0 silent, 1 all infos, 2 quiet infos.
    
    Returns:
    list: A list of random samples, one for each parameter combination.
    """
    groups = []
    for i, arg in enumerate(args):
        if len(arg) == 3:
            arg = list(arg) + ['linear'] # Add default method 'linear'
        
        # Ensure the good number of value for arg is pass
        if len(arg) != 4:
            raise AttributeError(f'All arguments must have a size of 3 or 4. Size is {len(arg)}.')
        
        bound_min, bound_max, n_values, method = arg
        
        # Ensure n_values is at least 2
        if n_values < 2:
            raise ValueError(f"n_values should be at least 2. Received: {n_values}")
        
        # Generate chunks based on the method (linear, log, uniform, gaussian, exponential, beta)
        if method in ('linear', 'uniform'):
            chunks = np.linspace(bound_min, bound_max, n_values + 1)
        elif method == 'log':
            chunks = np.logspace(np.log10(bound_min), np.log10(bound_max), n_values + 1)
        elif method == 'gaussian':
            # Gaussian sampling centered at mid point of the range
            mean = (bound_min + bound_max) / 2
            std_dev = (bound_max - bound_min) / 4  # Standard deviation: 1/4th of the range
            chunks = np.random.normal(loc=mean, scale=std_dev, size=n_values)
            chunks = np.clip(chunks, bound_min, bound_max)  # Ensure values are within bounds
        elif method == 'exponential':
            # Exponential sampling (biased towards the upper range)
            scale = (bound_max - bound_min) / 2
            chunks = bound_min + np.random.exponential(scale, size=n_values) * (bound_max - bound_min) / scale
            chunks = np.clip(chunks, bound_min, bound_max)  # Ensure values are within bounds
        elif method.startswith('beta'):
            # Beta distribution sampling (shaped by alpha and beta)
            alpha, beta = 2, 5  # Default values
            beta_args = beta.split('_')
            if len(beta_args) == 3:
                _, alpha, beta = beta_args
            elif len(beta_args) != 1:
                raise AttributeError(f'Beta method can only be \'beta\' or \'beta_<alphaVal>_<betaVal>.')
            chunks = np.random.beta(alpha, beta, size=n_values) * (bound_max - bound_min) + bound_min
        else:
            raise AttributeError(f"Method accepted are: 'linear', 'log', 'uniform', 'gaussian', 'exponential', 'beta'. Received: {method}")
                  
        chunks = np.column_stack((chunks[:-1], chunks[1:]))
                  
        if verbose == 1:
            print(f'chunks {i}: {chunks.shape} {chunks}')
            
        groups.append(chunks)
    
    # Generate all possible combinations of parameter values
    boxes =list(product(*groups))
    
    boxes_iterator = boxes if verbose == 0 else tqdm(boxes, total=len(boxes), mininterval=1.0)
    return [[np.random.uniform(bounds[0], bounds[1]) for bounds in boxe] for boxe in boxes_iterator]

            

import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import visualize

td1, xd1 = [], []
td2, xd2 = [], []
td3, xd3 = [], []
ns = 50_000 #1_000_000
step = ns // 100
n1, n2 = 0, 10000
xs = []
for i, n in enumerate(range(2, ns, step)):
    print(f'{i}/100', end='\r')
    x = n
    t = time.time()
    stratified((n1, n2, x))
    td1.append(time.time() - t)
    xd1.append(n)
    
    x = int(round(n ** (1. / 2)))
    if x>=2:
        t = time.time()
        stratified((n1, n2, x), (n1, n2, x))
        td2.append(time.time() - t)
        xd2.append(n)
    
    x = int(round(n ** (1. / 3)))
    if x>=2:
        t = time.time()
        stratified((n1, n2, x), (n1, n2, x), (n1, n2, x))
        td3.append(time.time() - t)
        xd3.append(n)

from matplotlib import pyplot as plt
visualize.curves((xd1, td1, {'label':'1D'}), (xd2, td2, {'label':'2D'}), (xd3, td3, {'label':'3D'}), show=True, legend=True, xlabel='Nombre de valeurs pour chaque variable', ylabel='Temps (s)', title='Performance du Grid Stratified Sampling selon la complexité')
# fig, ax1 = plt.subplots()
# ax1.tick_params(axis='y', labelcolor='blue')
# visualize.curves((xd1, td1, {'label':'1D', 'color':'blue'}), fig=fig, ax=ax1, show=False, xlabel='Nombre de valeurs pour chaque variable', ylabel='Temps (s)', title='Performance du Grid Stratified Sampling selon la complexité')
# ax2 = ax1.twinx()
# ax2.spines["left"].set_position(("outward", 60))
# ax2.tick_params(axis='y', labelcolor='orange')
# visualize.curves((xd2, td2, {'label':'2D', 'color':'orange'}),fig=fig, ax=ax2, show=False)
# ax3 = ax2.twinx()
# ax3.tick_params(axis='y', labelcolor='green')
# visualize.curves((xd3, td3, {'label':'3D', 'color':'green'}), fig=fig, ax=ax3, show=True, legend_fig=True)
# D, Z, n = 500, 500, 500
# values = np.array(stratified((25, 150, D), (1500, 3500, Z), (0.8, 1.2, n), verbose=2))

# values[:, :2] = np.round(values[:, :2])
# import pandas as pd
# df = pd.DataFrame(values)
# df.rename(columns={0:'D', 1:'Z', 2:'n'}, inplace=True)
# print(df)
# df.to_csv(rf'C:\Users\ET281306\Downloads\D{D}_Z{Z}_n{n}.csv', sep=';', encoding='utf-8-sig', index=False)