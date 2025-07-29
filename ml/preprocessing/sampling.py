import numpy as np
from itertools import product
from tqdm import tqdm
import random
from typing import Union

def _getChunks(bound_min, bound_max, n_values, method):
    """
    Generate chunk edges based on the specified method.
    Returns an array of shape (n_values, 2) where each row is [low, high].
    """
    if method in ('linear', 'uniform'):
        chunks = np.linspace(bound_min, bound_max, n_values + 1)
    elif method == 'log':
        chunks = np.logspace(np.log10(bound_min), np.log10(bound_max), n_values + 1)
    elif method == 'gaussian':
        mean = (bound_min + bound_max) / 2
        std_dev = (bound_max - bound_min) / 4
        chunks = np.random.normal(loc=mean, scale=std_dev, size=n_values+1)
        chunks = np.clip(chunks, bound_min, bound_max)
    elif method == 'exponential':
        scale = (bound_max - bound_min) / 2
        chunks = bound_min + np.random.exponential(scale, size=n_values+1) * (bound_max - bound_min) / scale
        chunks = np.clip(chunks, bound_min, bound_max)
    elif method.startswith('beta'):
        alpha, beta = 2, 5
        parts = method.split('_')
        if len(parts) == 3:
            _, alpha, beta = parts
            alpha, beta = float(alpha), float(beta)
        chunks = np.random.beta(alpha, beta, size=n_values+1) * (bound_max - bound_min) + bound_min
    else:
        raise AttributeError(
            f"Accepted methods: 'linear','uniform','log','gaussian','exponential','beta'. Received: {method}"
        )
    # pair up consecutive edges into (low, high) rows
    return np.column_stack((chunks[:-1], chunks[1:]))

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
        
        chunks = _getChunks(bound_min, bound_max, n_values, method)
                  
        if verbose == 1:
            print(f'chunks {i}: {chunks.shape} {chunks}')
            
        groups.append(chunks)
    
    # Generate all possible combinations of parameter values
    boxes =list(product(*groups))
    
    boxes_iterator = boxes if verbose == 0 else tqdm(boxes, total=len(boxes), mininterval=1.0)
    return [[np.random.uniform(bounds[0], bounds[1]) for bounds in boxe] for boxe in boxes_iterator]

         
def equilibrate(*args, method='max', fill_method='random', return_all=True, verbose=False):
    """
    Equilibrate a multi-dimensional dataset by binning and filling under-populated cells.

    Parameters
    ----------
    *args :
        Each argument must be one of:
          - (data_list, n_bins)
          - (data_list, bound_min, bound_max, n_bins)
          - (data_list, bound_min, bound_max, n_bins, chunk_method)
    method : {'max', 'mean'} or int, default='max'
        Target number of samples per occupied cell.
    fill_method : {'random', 'midpoint'}, default='random'
        How to generate new points within each under-populated cell.
    return_all : bool, default=True
        If True, return original + synthetic samples.
        If False, return only synthetic samples.
    verbose : bool, default=False
        If True, show a tqdm progress bar over all grid cells.

    Returns
    -------
    tuple of lists
        One list per dimension, containing either all samples or only the synthetic ones.
    """
    # Parse inputs and normalize to (array, bound_min, bound_max, n_bins, chunk_method)
    data_arrays = []
    bound_specs = []
    for arg in args:
        if len(arg) == 2:
            arr, n = arg
            lo, hi, cm = arr.min(), arr.max(), 'linear'
        elif len(arg) == 4:
            arr, lo, hi, n = arg
            cm = 'linear'
        elif len(arg) == 5:
            arr, lo, hi, n, cm = arg
        else:
            raise ValueError("Each argument must have 2, 4, or 5 elements")
        arr = np.asarray(arr)
        data_arrays.append(arr)
        bound_specs.append((lo, hi, int(n), cm))

    nb_dim = len(data_arrays)
    nb_datas = len(data_arrays[0])

    # Check all arrays have the same length
    for arr in data_arrays:
        if len(arr) != nb_datas:
            raise ValueError("All data arrays must have the same length")

    # Compute chunk edges for each dimension
    chunks_per_dim = [
        _getChunks(lo, hi, n_bins, cm)
        for lo, hi, n_bins, cm in bound_specs
    ]
    dims = [c.shape[0] for c in chunks_per_dim]  # number of bins per dimension

    # Digitize each original point into its bin index along each dimension
    bin_indices = []
    for d in range(nb_dim):
        edges = chunks_per_dim[d]  # shape (n_bins, 2)
        # find index i where value < edges[i,1]
        inds = np.searchsorted(edges[:,1], data_arrays[d], side='right')
        inds = np.clip(inds, 0, dims[d]-1)
        bin_indices.append(inds)

    # Count how many original points fall into each multi-dimensional cell
    flat_idx = np.ravel_multi_index(bin_indices, dims)
    counts = np.bincount(flat_idx, minlength=np.prod(dims)).reshape(dims)

    # Determine target count per occupied cell
    if isinstance(method, str):
        m = method.lower()
        if m == 'max':
            target = int(counts.max())
        elif m == 'mean':
            target = int(round(counts.mean()))
        else:
            raise ValueError("method must be one of 'max', 'mean', or an integer")
    else:
        target = int(method)

    # Generate synthetic samples for under-populated cells
    synthetic = [[] for _ in range(nb_dim)]
    cell_iterator = product(*[range(n) for n in dims])
    if verbose:
        cell_iterator = tqdm(list(cell_iterator), desc="Equilibrating cells")

    for cell in cell_iterator:
        current = counts[cell]
        deficit = target - current
        if deficit <= 0:
            continue

        # compute the hyper-rectangle bounds for this cell
        lows  = [chunks_per_dim[d][cell[d], 0] for d in range(nb_dim)]
        highs = [chunks_per_dim[d][cell[d], 1] for d in range(nb_dim)]

        # generate the missing points
        for _ in range(deficit):
            if fill_method == 'random':
                point = [np.random.uniform(lo, hi) for lo, hi in zip(lows, highs)]
            elif fill_method == 'midpoint':
                point = [(lo + hi) / 2.0 for lo, hi in zip(lows, highs)]
            else:
                raise ValueError("fill_method must be 'random' or 'midpoint'")

            for d, val in enumerate(point):
                synthetic[d].append(val)

    # Prepare outputs
    outputs = []
    for d in range(nb_dim):
        if return_all:
            outputs.append(list(data_arrays[d]) + synthetic[d])
        else:
            outputs.append(synthetic[d])

    return tuple(outputs)