import numpy as np

def meanAbsoluteErrorNormalised(true, pred, div_0=1e-10):  
    true, pred = np.array(true), np.array(pred)
    true_nonzero = np.where(true == 0, div_0, true)
    return np.sum(np.abs(true - pred) / np.abs(true_nonzero)) / len(true)
   