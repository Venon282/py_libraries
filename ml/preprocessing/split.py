import numpy as np
from sklearn.model_selection import train_test_split

def indice(indices, mask=None, split_ratios=(0.7, 0.15, 0.15), seed=42):
    # Apply Mask 
    if mask:
        valid_indices = indices[mask.astype(bool)]
    else:
        valid_indices = indices
        
    # Perform Splitting on the INDICES only
    # First split: Train vs (Val + Test)
    train_idx, temp_idx = train_test_split(
        valid_indices, 
        test_size=(1 - split_ratios[0]), 
        random_state=seed, 
        shuffle=True
    )
    
    # Second split: Val vs Test
    # Normalize the test size relative to the remaining data
    relative_test_size = split_ratios[2] / (split_ratios[1] + split_ratios[2])
    val_idx, test_idx = train_test_split(
        temp_idx, 
        test_size=relative_test_size, 
        random_state=seed, 
        shuffle=True
    )
    
    return {'train': train_idx, 'val': val_idx, 'test': test_idx} 
        
    

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