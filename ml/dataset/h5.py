import tensorflow as tf
import h5py
import logging
import math
logger = logging.getLogger(__name__)

class H5Generator:
    def __init__(self, h5_path, indices, input_cols, output_cols):
        self.h5_path = h5_path
        self.indices = indices
        self.input_cols = input_cols
        self.output_cols = output_cols
        
    def __call__(self):
        logger.debug(f"Creating generator for {len(self.indices)} indices")
        with h5py.File(self.h5_path, 'r') as f:
            for idx in self.indices:
                # Read Inputs
                # Returns a dict {col_name: data} 
                inputs = {
                    col: f[col][idx] for col in self.input_cols
                }
                
                # Read Outputs
                # Can be a dict or a single array/value depending on the loss function
                if len(self.output_cols) == 1:
                    outputs = f[self.output_cols[0]][idx]
                else:
                    outputs = {col: f[col][idx] for col in self.output_cols}
                
                yield inputs, outputs
        logger.debug (f"exhausted for {len(self.indices)} indices.")
        

                
def makeTfDataset(h5_path, indices, input_cols, output_cols, batch_size=32, return_steps_per_epoch=False, epochs=1, seed=42):
    """
    Creates a tf.data.Dataset
    """
    def stepsFor(n_samples, batch_size):
        return math.ceil(n_samples / batch_size) if n_samples > 0 else 0
    
    # Get shapes and dtypes for signature
    with h5py.File(h5_path, 'r') as f:
        input_signatures = {
            col: tf.TensorSpec(shape=f[col].shape[1:], dtype=f[col].dtype)
            for col in input_cols
        }
        
        if len(output_cols) == 1:
            output_signatures = tf.TensorSpec(shape=f[output_cols[0]].shape[1:], dtype=f[output_cols[0]].dtype)
        else:
            output_signatures = {
                col: tf.TensorSpec(shape=f[col].shape[1:], dtype=f[col].dtype)
                for col in output_cols
            }
            
    signature = (input_signatures, output_signatures)

    # Instantiate generator
    gen_obj = H5Generator(h5_path, indices, input_cols, output_cols)

    # Create Dataset
    ds = tf.data.Dataset.from_generator(
        gen_obj,
        output_signature=signature
    )

    # Pipeline Optimizations
    ds = ds.shuffle(buffer_size=min(len(indices), 10 * batch_size), seed=seed) 
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.repeat(epochs)
    
    if return_steps_per_epoch:
        return ds, stepsFor(len(indices), batch_size)
    
    return ds

"""
Exemple:
indices_map = splitH5(h5_path, mask=mask, split_ratios=cfgt['split_ratios'], seed=cfgt.get('seed', 42)) 

# Create TF Datasets
train_ds, train_steps = makeTfDataset(h5_path, indices_map['train'], input_cols, output_cols, batch_size=batch_size, epochs=epochs, return_steps_per_epoch=True)
val_ds, val_steps = makeTfDataset(h5_path, indices_map['val'], input_cols, output_cols, batch_size=batch_size, epochs=epochs, return_steps_per_epoch=True)
test_ds, test_steps = makeTfDataset(h5_path, indices_map['test'], input_cols, output_cols, batch_size=batch_size, epochs=1, return_steps_per_epoch=True)

history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks, 
        verbose=cfgt['verbose']
    )
"""