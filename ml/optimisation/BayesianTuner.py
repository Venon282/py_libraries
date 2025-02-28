import keras_tuner as kt # type: ignore
import sklearn.gaussian_process as gp # type: ignore
from scipy.optimize import fmin_l_bfgs_b
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore

class BayesianTuner(kt.BayesianOptimization):
    
    def __init__(self, *args, **kwargs):
        super(BayesianTuner, self).__init__(*args, **kwargs)
        
        self.length_scale_min = kwargs.get('length_scale_min', 1e-08)
        self.length_scale_max = kwargs.get('length_scale_max', 1e5)
        self.optimizer_maxiter = kwargs.get('optimizer_maxiter', 25000)
        
        self.batch_size_min = kwargs.get('batch_size_min', 32)
        self.batch_size_max = kwargs.get('batch_size_max', 2048)
        self.batch_size_step = kwargs.get('batch_size_step', 1)
        
        self.early_stopping_patience_min = kwargs.get('early_stopping_patience_min', 30)
        self.early_stopping_patience_max = kwargs.get('early_stopping_patience_max', 70)
        self.early_stopping_patience_step = kwargs.get('early_stopping_patience_step', 10)
        
        self.rlrop_factor_min = kwargs.get('rlrop_factor_min', 0.1)
        self.rlrop_factor_max = kwargs.get('rlrop_factor_max', 0.9)
        self.rlrop_factor_step = kwargs.get('rlrop_factor_step', 0.1)
        
        self.rlrop_patience_min = kwargs.get('rlrop_patience_min', 5)
        self.rlrop_patience_max = kwargs.get('rlrop_patience_max', 20)
        self.rlrop_patience_step = kwargs.get('rlrop_patience_step', 1)
        
        self.min_lr = kwargs.get('min_lr', 1e-8)
        self.verbose = kwargs.get('verbose', 0) # Remove display at each epochs
    
    # def _build_default_model(self):
    #     # Adjust the RBF kernel's lower bound from 1e-05 to 1e-08.
    #     kernel = gp.kernels.ConstantKernel(1.0, constant_value_bounds="fixed") * gp.kernels.RBF(
    #         length_scale=1.0, 
    #         length_scale_bounds=(self.length_scale_min, self.length_scale_max)
    #     )
    #     normalize_y = getattr(self._oracle, "normalize_y", True)
    #     model = gp.GaussianProcessRegressor(
    #         kernel=kernel,
    #         alpha=self._oracle.alpha,
    #         normalize_y=normalize_y,
    #         random_state=self._oracle.seed,
    #         optimizer= lambda obj_func, initial_theta, bounds: fmin_l_bfgs_b(obj_func, initial_theta, approx_grad=True, bounds=bounds, maxiter=self.optimizer_maxiter),
    #         n_restarts_optimizer=20,
    #     )
    #     return model

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        
        batch_size = hp.Int('batch_size', min_value=self.batch_size_min, max_value=self.batch_size_max, step=self.batch_size_step)
        es_patience = hp.Int('early_stopping_patience', min_value=self.early_stopping_patience_min, max_value=self.early_stopping_patience_max, step=self.early_stopping_patience_step)
        callbacks = [
            EarlyStopping(monitor='val_loss', 
                          patience=es_patience, 
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', 
                              factor=hp.Float('rlrop_factor', min_value=self.rlrop_factor_min, max_value=self.rlrop_factor_max, step=self.rlrop_factor_step), 
                              patience=hp.Int('rlrop_patience', min_value=self.rlrop_patience_min, max_value=min(es_patience, self.rlrop_patience_max), step=self.rlrop_patience_step), 
                              min_lr=self.min_lr)
        ]
        
        kwargs['callbacks'] = callbacks
        kwargs['batch_size'] = batch_size
        kwargs['verbose'] = self.verbose # 0: silent, 1: progress bar, 2: one line per epoch
        result = super(BayesianTuner, self).run_trial(trial, *args, **kwargs)
        return result