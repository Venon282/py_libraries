import numpy as np
from itertools import product
from tqdm import tqdm
import random
from typing import Union
import logging

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

         
def equilibrate(*args, method='max', fill_method='random', return_all=True,
                trim_bound_excess=False, trim_cell_excess=False, verbose=False, **kwargs):
    """
    Equilibrate a multi-dimensional dataset by binning and filling under-populated cells.

    Parameters
    ----------
    *args :
        Each argument must be one of:
          - (data_list, n_bins)
          - (data_list, n_bins, chunk_method)
          - (data_list, bound_min, bound_max, n_bins)
          - (data_list, bound_min, bound_max, n_bins, chunk_method)
          chunk_method can be  'linear','uniform','log','gaussian','exponential','beta'. Default is linear.
    method : {'max', 'mean'} or int, default='max'
        Target number of samples per occupied cell.
    fill_method : {'random', 'midpoint', 'gap'}, default='random'
        How to generate new points within each under-populated cell.
        - random: random point in the cell
        - midpoint: point in the center of the cell
        - linear_gap: point in the most empty space on each dimension
        - dimensional_gap: point in the most empty space of the cell based on subpoints (costly)
            - n_sub (default 3) give n_sub^n_dims candidate points.
        
    return_all : bool, default=True
        If True, return original + synthetic samples.
        If False, return only synthetic samples.
    trim_bound_excess: bool, default=False
        If true, the original array will be trim based on bounds defined
    trim_cell_excess : bool, default=False
        If True, randomly downsample points in over-populated cells to match `method`. Process only when return_all is true
    verbose : bool, default=False
        If True, show a tqdm progress bar over all grid cells.

    Returns
    -------
    tuple of lists or tuple of tuples of lists
        If `return_all=True` (default), returns a tuple of three tuples:
            - combined : tuple of lists
                One list per dimension containing original + synthetic samples.
            - original : tuple of lists
                One list per dimension containing only the original samples (after trimming if applied).
            - synthetic : tuple of lists
                One list per dimension containing only the synthetic samples generated to fill under-populated cells.
        If `return_all=False`, returns:
            - synthetic : tuple of lists
                One list per dimension containing only the synthetic samples.
    """
    if fill_method not in ('random', 'midpoint', 'dimensional_gap', 'linear_gap'):
        raise ValueError("fill_method must be 'random', 'midpoint', 'dimensional_gap' or 'linear_gap'")
    
    # Parse inputs and normalize to (array, bound_min, bound_max, n_bins, chunk_method)
    data_arrays = []
    bound_specs = []
    mask = np.full_like(np.asarray(args[0][0]), True, dtype=bool)
    for arg in args:
        if len(arg) == 2:
            arr, n = arg
            lo, hi, cm = np.asarray(arr).min(), np.asarray(arr).max(), 'linear'
        elif len(arg) == 3:
            arr, n, cm = arg
            lo, hi = np.asarray(arr).min(), np.asarray(arr).max()
        elif len(arg) == 4:
            arr, lo, hi, n = arg
            cm = 'linear'
        elif len(arg) == 5:
            arr, lo, hi, n, cm = arg
        else:
            raise ValueError("Each argument must have 2, 3, 4, or 5 elements")
        arr = np.asarray(arr)
        if trim_bound_excess:
            mask = mask & (arr >= lo) & (arr <= hi)
        data_arrays.append(arr)
        bound_specs.append((lo, hi, int(n), cm))
    if trim_bound_excess:
        data_arrays = [arr[mask] for arr in data_arrays]
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
        inds = np.searchsorted(edges[:, 1], data_arrays[d], side='right')
        inds = np.clip(inds, 0, dims[d] - 1)
        bin_indices.append(inds)

    # Count how many original points fall into each multi-dimensional cell
    flat_idx = np.ravel_multi_index(tuple(bin_indices), dims)
    counts = np.bincount(flat_idx, minlength=int(np.prod(dims))).reshape(dims)

    # Determine target count per occupied cell
    if isinstance(method, str):
        m = method.lower()
        if m == 'max':
            target = int(counts.max())
        elif m == 'mean':
            target = int(round(counts.mean()))
        else:
            raise ValueError("method must be one of 'max', 'mean', or an integer")
        logging.debug(f'Number of values by cells: {target}')
    else:
        target = int(method)

    # Generate synthetic samples for under-populated cells
    synthetic = [[] for _ in range(nb_dim)]
    cell_iterator = product(*[range(n) for n in dims])
    if verbose:
        try:
            from tqdm import tqdm
            cell_iterator = tqdm(list(cell_iterator), desc="Equilibrating cells", mininterval=1)
        except Exception:
            cell_iterator = list(cell_iterator)

    # Track which original points to keep
    keep_mask = np.ones(nb_datas, dtype=bool)

    for cell in cell_iterator:
        current = counts[cell]
        deficit = target - current

        # compute indices of original points inside this cell (always compute)
        # build boolean array that is True when point belongs to the given cell in all dims
        cell_members_bool = np.ones(nb_datas, dtype=bool)
        for d in range(nb_dim):
            cell_members_bool &= (bin_indices[d] == cell[d])
        idx_in_cell = np.where(cell_members_bool)[0]

        if deficit < 0 and trim_cell_excess and return_all:
            # Random downsampling to target
            if idx_in_cell.size > target:
                keep = np.random.choice(idx_in_cell, size=target, replace=False)
                remove = np.setdiff1d(idx_in_cell, keep)
                keep_mask[remove] = False
            continue

        if deficit <= 0:
            continue

        # compute the hyper-rectangle bounds for this cell
        lows = [chunks_per_dim[d][cell[d], 0] for d in range(nb_dim)]
        highs = [chunks_per_dim[d][cell[d], 1] for d in range(nb_dim)]

        # Prepare current points in the cell for gap calculation
        current_points = [list(data_arrays[d][idx_in_cell]) for d in range(nb_dim)]

        # generate the missing points
        for _ in range(deficit):
            if fill_method == 'random':
                point = [np.random.uniform(lo, hi) for lo, hi in zip(lows, highs)]
            elif fill_method == 'midpoint':
                point = [(lo + hi) / 2.0 for lo, hi in zip(lows, highs)]
            elif fill_method == 'linear_gap':
                point = []
                for d in range(nb_dim):
                    existing = np.asarray(current_points[d])
                    if existing.size == 0:
                        # No points in the cell → fallback to midpoint
                        val = (lows[d] + highs[d]) / 2.0
                    else:
                        existing_sorted = np.sort(existing)
                        # Include cell boundaries
                        extended = np.concatenate(([lows[d]], existing_sorted, [highs[d]]))
                        # Find largest gap
                        gaps = np.diff(extended)
                        max_gap_idx = np.argmax(gaps)
                        # Midpoint of the largest gap
                        val = (extended[max_gap_idx] + extended[max_gap_idx + 1]) / 2.0
                    point.append(val)
            elif fill_method == 'dimensional_gap':
                # If the cell is empty, place midpoint
                if len(current_points[0]) == 0:
                    point = [(lo + hi) / 2.0 for lo, hi in zip(lows, highs)]
                else:
                    n_sub = int(kwargs.get('n_sub', 3))
                    # create grid of candidate points inside the cell (exclude boundaries)
                    grids = [np.linspace(lo, hi, n_sub + 2)[1:-1] for lo, hi in zip(lows, highs)]
                    candidates = np.array(list(product(*grids)))
                    existing = np.column_stack(current_points)
                    from scipy.spatial import cKDTree
                    tree = cKDTree(existing)
                    dists, _ = tree.query(candidates)
                    point = candidates[np.argmax(dists)]

            # append generated point to current_points and synthetic lists
            for d, val in enumerate(point):
                current_points[d].append(val)
                synthetic[d].append(val)

    # Convert data to arrays
    data_original = np.stack([np.asarray(arr)[keep_mask] for arr in data_arrays], axis=1)  # shape (n_samples, nb_dim)

    # Handle case of zero synthetic points
    if any(len(s) > 0 for s in synthetic):
        data_synthetic = np.stack([np.asarray(arr) for arr in synthetic], axis=1)            # shape (n_synthetic, nb_dim)
    else:
        data_synthetic = np.empty((0, nb_dim))

    if return_all:
        if data_synthetic.size == 0:
            data_combined = data_original.copy()
        else:
            data_combined = np.vstack([data_original, data_synthetic])
        return data_combined.T.tolist(), data_original.T.tolist(), data_synthetic.T.tolist()
    else:
        return data_synthetic.T.tolist()
    
