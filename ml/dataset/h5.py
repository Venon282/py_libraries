import tensorflow as tf
import h5py
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

def loadFromH5(h5_path, index, input_cols, output_cols):
    """
    Core logic to extract a single sample from HDF5.
    This runs inside a tf.numpy_function, allowing for parallel disk I/O.
    """
    # Ensure arguments are converted from tensor bytes to native strings/ints
    path = h5_path.decode('utf-8') if isinstance(h5_path, bytes) else h5_path
    idx = int(index)
    
    # Handle list of columns that might be passed as byte tensors
    in_cols = [c.decode('utf-8') if isinstance(c, bytes) else c for c in input_cols]
    out_cols = [c.decode('utf-8') if isinstance(c, bytes) else c for c in output_cols]

    with h5py.File(path, 'r') as f:
        # Extract input features as a flat list
        input_data = [f[col][idx] for col in in_cols]
        
        # Extract target outputs
        if len(out_cols) == 1:
            output_data = [f[out_cols[0]][idx]]
        else:
            output_data = [f[col][idx] for col in out_cols]

    # Return a flat tuple as required by tf.numpy_function
    return tuple(input_data + output_data)

def makeTfDataset(
    h5_path: str, 
    indices: list, 
    input_cols: list, 
    output_cols: list, 
    batch_size: int = 32, 
    epochs: int | None = 1, 
    shuffle: bool = True,
    deterministic: bool = None,
    return_steps_per_epoch: bool = False, 
    seed: int = 42
):
    """
    Constructs a high-performance tf.data pipeline using parallel mapping.
    """
    
    # Inspect HDF5 metadata to establish shapes and dtypes for the TF graph
    with h5py.File(h5_path, 'r') as f:
        input_dtypes = [f[col].dtype for col in input_cols]
        input_shapes = [f[col].shape[1:] for col in input_cols]
        output_dtypes = [f[col].dtype for col in output_cols]
        output_shapes = [f[col].shape[1:] for col in output_cols]

    all_dtypes = input_dtypes + output_dtypes
    
    # Initialize dataset with indices to minimize memory overhead during shuffle
    dataset = tf.data.Dataset.from_tensor_slices(indices)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(indices), seed=seed)

    def mapFunction(index):
        """
        Wraps the numpy loader and restores tensor metadata lost during numpy conversion.
        """
        # Execute the python-based loader in parallel
        result = tf.numpy_function(
            func=loadFromH5,
            inp=[h5_path, index, input_cols, output_cols],
            Tout=all_dtypes
        )
        
        # Reconstruct the feature dictionary and apply shapes
        inputs = {}
        for i, col in enumerate(input_cols):
            tensor = result[i]
            tensor.set_shape(input_shapes[i])
            inputs[col] = tensor
            
        # Reconstruct the label structure (single tensor or dictionary)
        offset = len(input_cols)
        if len(output_cols) == 1:
            outputs = result[offset]
            outputs.set_shape(output_shapes[0])
        else:
            outputs = {}
            for i, col in enumerate(output_cols):
                tensor = result[offset + i]
                tensor.set_shape(output_shapes[i])
                outputs[col] = tensor
                
        return inputs, outputs

    # Map the loading function with multiple CPU threads
    # If need to be deterministic, impossible to use multiple CPU
    """ doc
    Performance can often be improved by setting num_parallel_calls so that map will use multiple threads to process elements. 
    If deterministic order isn't required, it can also improve performance to set deterministic=False.
    The order of elements yielded by this transformation is deterministic if deterministic=True. 
    If map_func contains stateful operations and num_parallel_calls > 1, the order in which that state is accessed is undefined, so the values of output elements may not be deterministic regardless of the deterministic flag value.
    """
    # tf.data.AUTOTUNE if deterministic is False else 1
    dataset = dataset.map(mapFunction, num_parallel_calls=tf.data.AUTOTUNE, deterministic=deterministic)

    # Finalize the pipeline with batching and background prefetching
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    if return_steps_per_epoch:
        steps = math.ceil(len(indices) / batch_size) if len(indices) > 0 else 0
        return dataset, steps
    
    return dataset