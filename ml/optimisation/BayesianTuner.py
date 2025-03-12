import keras_tuner as kt # type: ignore
import sklearn.gaussian_process as gp # type: ignore
from scipy.optimize import fmin_l_bfgs_b
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from .module import setHyperparameter

class BayesianTuner(kt.BayesianOptimization):
    
    def __init__(self, *args, fixe_hparams={}, **kwargs):
        self.fixe_hparams = fixe_hparams
        
        self.length_scale_min = kwargs.pop('length_scale_min', 1e-08)
        self.length_scale_max = kwargs.pop('length_scale_max', 1e5)
        self.optimizer_maxiter = kwargs.pop('optimizer_maxiter', 25000)
        
        self.batch_size_min = kwargs.pop('batch_size_min', 32)
        self.batch_size_max = kwargs.pop('batch_size_max', 2048)
        self.batch_size_step = kwargs.pop('batch_size_step', 1)
        
        self.early_stopping_patience_min = kwargs.pop('early_stopping_patience_min', 30)
        self.early_stopping_patience_max = kwargs.pop('early_stopping_patience_max', 70)
        self.early_stopping_patience_step = kwargs.pop('early_stopping_patience_step', 10)
        
        self.rlrop_factor_min = kwargs.pop('rlrop_factor_min', 0.1)
        self.rlrop_factor_max = kwargs.pop('rlrop_factor_max', 0.9)
        self.rlrop_factor_step = kwargs.pop('rlrop_factor_step', 0.1)
        
        self.rlrop_patience_min = kwargs.pop('rlrop_patience_min', 5)
        self.rlrop_patience_max = kwargs.pop('rlrop_patience_max', 20)
        self.rlrop_patience_step = kwargs.pop('rlrop_patience_step', 1)
        
        self.min_lr = kwargs.pop('min_lr', 1e-8)
        self.verbose = kwargs.pop('verbose', 0) # Remove display at each epochs
        
        super(BayesianTuner, self).__init__(*args, **kwargs)


    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        
        batch_size = setHyperparameter(hp.Int, self.fixe_hparams, 'batch_size', min_value=self.batch_size_min, max_value=self.batch_size_max, step=self.batch_size_step)
        es_patience = setHyperparameter(hp.Int, self.fixe_hparams, 'early_stopping_patience', min_value=self.early_stopping_patience_min, max_value=self.early_stopping_patience_max, step=self.early_stopping_patience_step)
        callbacks = [
            EarlyStopping(monitor='val_loss', 
                          patience=es_patience, 
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', 
                              factor=setHyperparameter(hp.Float, self.fixe_hparams, 'rlrop_factor', min_value=self.rlrop_factor_min, max_value=self.rlrop_factor_max, step=self.rlrop_factor_step), 
                              patience=setHyperparameter(hp.Int, self.fixe_hparams, 'rlrop_patience', min_value=self.rlrop_patience_min, max_value=min(es_patience, self.rlrop_patience_max), step=self.rlrop_patience_step), 
                              min_lr=self.min_lr)
        ]
        
        kwargs['callbacks'] = callbacks
        kwargs['batch_size'] = batch_size
        kwargs['verbose'] = self.verbose # 0: silent, 1: progress bar, 2: one line per epoch
        result = super(BayesianTuner, self).run_trial(trial, *args, **kwargs)
        return result