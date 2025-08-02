import numpy as np

def metrics(true, pred, eps=1e-12):
    # 1) Core differences
    diff        = pred - true
    abs_diff    = np.abs(diff)
    abs_true    = np.abs(true)
    abs_pred    = np.abs(pred)
    sq_diff     = diff**2

    # 2) Basic stats
    n           = true.shape[0]
    mae         = abs_diff.mean()
    mse         = sq_diff.mean()
    rmse        = np.sqrt(mse)

    # 3) Normalized MAE (example: MAE divided by mean(true))
    maen        = mae / (abs_true.mean() + eps)

    # 4) sMAPE: avoid zero-division by adding tiny eps
    denom       = abs_true + abs_pred + eps
    smape       = (2 * abs_diff / denom).mean()

    # 5) R² score from first principles
    ss_res      = sq_diff.sum()
    ss_tot      = ((true - true.mean())**2).sum()
    r2          = 1 - ss_res/ss_tot

    return {
        'mae'   : mae,
        'maen'  : maen,
        'smape' : smape,
        'mse'   : mse,
        'rmse'  : rmse,
        'r2'    : r2
}