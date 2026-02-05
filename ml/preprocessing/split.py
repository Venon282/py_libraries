import numpy as np
from sklearn.model_selection import train_test_split

def indice(indices, mask=None, split_ratios=(0.7, 0.15, 0.15), seed=42, names=['train', 'val', 'test']):
    # Apply Mask 
    if mask:
        valid_indices = indices[mask.astype(bool)]
    else:
        valid_indices = indices
    
    # Verify the split_ratio tuple
    total_sum = sum(split_ratios)
    if total_sum < 1.0:
        split_ratios = (*split_ratios, 1.0 - total_sum)
    elif total_sum > 1.0:
        raise AttributeError('The split_ratios can\'t be superior to 1.')
    
    if len(split_ratios) == 1:
        return {names[0] if names else '0': valid_indices}
    
    indice_dict = {}                    # Dict containing the results
    remaining_indices = valid_indices   
    remaining_ratio = 1.0
    for i in range(len(split_ratios)-1):
        # Get the name
        try:
            name = names[i]
        except:
            name = str(i)
            
        if split_ratios[i] == 0.0:
            indice_dict[name] = np.array([], dtype=int)
            continue
        
        test_size = 1 - (split_ratios[i] / remaining_ratio)
        remaining_ratio -= split_ratios[i] 
        
        current_set_indices, remaining_indices = train_test_split(
            remaining_indices, 
            test_size=test_size, 
            random_state=seed, 
            shuffle=True
        )
        
        indice_dict[name] = current_set_indices
    
    # Insert the last set
    try:
        name = names[i+1]
    except:
        name = str(i+1)
    indice_dict[name] = remaining_indices
    
    return indice_dict
        
    

def h5(h5_path, mask=None, split_ratios=(0.7, 0.15, 0.15), seed=42):
    """
    Returns a dictionary containing 'train', 'val', 'test' arrays of INDICES.
    """
    import h5py
    
    with h5py.File(h5_path, 'r') as f:
        # We assume all columns have the same length, pick one to check
        first_key = next(iter(f.keys()))
        total_rows = f[first_key].shape[0]
        all_indices = np.arange(total_rows)
        
    return indice(all_indices, mask=mask, split_ratios=split_ratios, seed=seed)