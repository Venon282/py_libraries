import sklearn.metrics as sklm
import numpy as np 

def metrics(true, pred):
    """
    Calculate classification metrics for softmax model predictions.

    Args:
        true (array-like): True class labels. If provided as one-hot encoded, they will be converted.
        pred (array-like): Predicted probabilities for each class (from the softmax layer).

    Returns:
        dict: A dictionary containing:
            - 'accuracy': Accuracy score.
            - 'log_loss': Logarithmic loss (cross entropy).
            - 'precision': Weighted precision score.
            - 'recall': Weighted recall score.
            - 'f1': Weighted F1 score.
    """
    # Convert one-hot encoded true labels to class labels, if necessary.
    if np.ndim(true) > 1:
        true = np.argmax(true, axis=1)
    
    # Convert predicted probabilities to class predictions.
    pred_class = np.argmax(pred, axis=1) if np.ndim(pred) > 1 else pred

    return {
        'accuracy': sklm.accuracy_score(true, pred_class),
        'log_loss': sklm.log_loss(true, pred),
        'precision': sklm.precision_score(true, pred_class, average='weighted', zero_division=0),
        'recall': sklm.recall_score(true, pred_class, average='weighted', zero_division=0),
        'f1': sklm.f1_score(true, pred_class, average='weighted', zero_division=0)
    }