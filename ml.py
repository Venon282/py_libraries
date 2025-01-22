import numpy as np

def meanAbsoluteErrorNormalised(true, pred):  
    true, pred = np.array(true), np.array(pred)
    return np.sum(np.abs(true - pred) / true) / len(true)