import tensorflow.keras as keras # type: ignore
from .WeightedAverage import WeightedAverage


def merge(mode, name=''):
    if mode == 'concat':
        return keras.layers.Concatenate(name=f'{name}_concat')
    elif mode == 'add':
        return keras.layers.Add(name=f'{name}_add')
    elif mode == 'average':
        return keras.layers.Average(name=f'{name}_average')
    elif mode == 'weighted_avg':
        return WeightedAverage(name=f'{name}_weighted_average')
    else:
        raise ValueError(f'Invalid merge mode: {mode}')
    
def activation(activation, negative_slope=0.3):
    if activation == 'leaky_relu':
        return keras.layers.LeakyReLU(negative_slope=negative_slope)
    elif activation == 'prelu':
        return keras.layers.PReLU()
    else:
        return keras.layers.Activation(keras.activations.get(activation))
    
def residualConnection(l1, l2):
    # Project x_in if necessary to match dimensions.
    if l1.shape[-1] != l2.shape[-1]:
       l1 = keras.layers.Dense(l2.shape[-1], activation=None)(l1)
    return keras.layers.Add()([l2, l1])