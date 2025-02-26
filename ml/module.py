import tensorflow.keras as keras # type: ignore

# Helper method to return a regularizer given the choice.
def getRegularizers(mode, l1=1e-4, l2=1e-4):
    if mode == 'l1':
        return keras.regularizers.l1(l1)
    elif mode == 'l2':
        return keras.regularizers.l2(l2)
    elif mode == 'l1_l2':
        return keras.regularizers.l1_l2(l1, l2)
    else:
        raise ValueError(f"Unknown regularizer choice: {mode}")
    
def getOptimizers(mode, lr=1e-3):
    if mode == 'adam':  
        return keras.optimizers.Adam(learning_rate=lr)
    elif mode == 'sgd':
        return keras.optimizers.SGD(learning_rate=lr)
    elif mode == 'rmsprop':  # rmsprop
        return keras.optimizers.RMSprop(learning_rate=lr)
    else:
        raise ValueError(f'Invalid optimizer choice: {mode}')
    
