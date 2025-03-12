import sklearn.metrics as sklm
from .module import meanAbsoluteErrorNormalised, symmetricMeanAbsolutePercentageError

def metrics(true, pred):
   return {
       'mae'    : sklm.mean_absolute_error(true, pred),
       'maen'   : meanAbsoluteErrorNormalised(true, pred),
       'smape'  : symmetricMeanAbsolutePercentageError(true, pred),
       'mse'    : sklm.mean_squared_error(true, pred),
       'r2'     : sklm.r2_score(true, pred),
       'rmse'   :sklm.root_mean_squared_error(true, pred)
   }