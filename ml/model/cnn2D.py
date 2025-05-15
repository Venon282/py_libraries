from typing import Tuple, List, Dict, Union, Optional
from tensorflow.keras import layers, Model # type: ignore

#! bad, delete it

def outputNames(units, activation):
    if isinstance(units, list):
        return units
    elif isinstance(units, int):
        return [f"output_{activation}_{i}" for i in range(units)]
    else:
        raise ValueError("units must be int or list")

def build_cnn(
    input_shape: Tuple[int, int, int],
    conv_blocks: List[Dict[str, Union[int, Tuple[int, int], str, bool, Tuple[int, int], float]]],
    dense_blocks: Optional[List[Dict[str, Union[int, str, float]]]] = None,
    transition: str = 'GlobalAveragePooling2D',
    output_specs: Dict[str, int] = {'softmax': 10}
) -> Model:
    """
    Builds a configurable 2D CNN using Keras Functional API.

    Args:
        input_shape: Shape of the input tensor (H, W, C).
        conv_blocks: List of dictionaries, each specifying a conv block:
            - filters (int): Number of convolution filters.
            - kernel_size (tuple): Kernel size, e.g. (3,3).
            - activation (str): Activation after convolution (default 'relu').
            - padding (str): Padding mode (default 'same').
            - batch_norm (bool): Whether to apply BatchNormalization (default False).
            - pool (str): Pooling layer name, e.g. 'MaxPooling2D' (optional).
            - pool_size (tuple): Pool size for pooling (default (2,2)).
            - dropout (float): Dropout rate after block (optional).
        dense_blocks: List of dictionaries for dense layers:
            - units (int): Number of neurons.
            - activation (str): Activation for the dense layer.
            - dropout (float): Dropout rate after dense (optional).
        output_specs: Mapping from activation name to number of outputs.

    Returns:
        Keras Model instance.
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Convolutional blocks
    for block in conv_blocks:
        filters = block['filters']
        kernel_size = block.get('kernel_size', (3, 3))
        padding = block.get('padding', 'same')
        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)

        if block.get('batch_norm', False):
            x = layers.BatchNormalization()(x)

        activation = block.get('activation', 'relu')
        x = layers.Activation(activation)(x)

        pool = block.get('pool')
        if pool:
            PoolLayer = getattr(layers, pool)
            pool_size = block.get('pool_size', (2, 2))
            x = PoolLayer(pool_size=pool_size)(x)

        dropout_rate = block.get('dropout', 0)
        if dropout_rate and dropout_rate > 0:
            x = layers.SpatialDropout2D(dropout_rate)(x)

    # Flatten before dense layers
    x = getattr(layers, transition)()(x)

    # Dense blocks
    if dense_blocks:
        for db in dense_blocks:
            units = db['units']
            activation = db.get('activation', 'relu')
            x = layers.Dense(units, activation=activation)(x)

            dropout_rate = db.get('dropout', 0)
            if dropout_rate and dropout_rate > 0:
                x = layers.Dropout(dropout_rate)(x)

    # Output layers
    outputs = []
    for act_name, units in output_specs.items():
        if act_name not in ['softmax', 'sigmoid', 'linear']:
            raise ValueError(f"Unsupported activation: {act_name}")

        for name in outputNames(units, activation=act_name):
            outputs.append(
                layers.Dense(1, activation=act_name, name=name)(x)
            )

    model = Model(inputs=inputs, outputs=outputs)
    return model


# Example usage:
# conv_cfg = [
#     {'filters': 32, 'kernel_size': (3,3), 'activation': 'relu', 'batch_norm': True, 'pool': 'MaxPooling2D', 'dropout': 0.2},
#     {'filters': 64, 'kernel_size': (3,3), 'activation': 'relu', 'batch_norm': True, 'pool': 'MaxPooling2D', 'dropout': 0.3},
# ]
# dense_cfg = [
#     {'units': 256, 'activation': 'relu', 'dropout': 0.5},
# ]
# model = build_cnn((128,128,3), conv_cfg, dense_cfg, {'softmax': 10})
# model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.summary()
