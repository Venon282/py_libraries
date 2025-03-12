import numpy as np

def meanAbsoluteErrorNormalised(true, pred, eps=1e-10):  
    true, pred = np.array(true), np.array(pred)
    denom = np.maximum(np.abs(true), eps)
    return np.mean(np.abs(true - pred) / denom)

def symmetricMeanAbsolutePercentageError(true, pred, eps=1e-10):
    # Symmetric mean absolute percentage error
    true, pred = np.array(true), np.array(pred)
    return np.mean(2 * np.abs(true - pred) / (np.abs(true) + np.abs(pred) + eps))
