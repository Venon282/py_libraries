import tensorflow as tf

from .ModelBuilder import BaseModelBuilder
from .module import setHyperparameter
from ..model import TransformerForecaster
from ..optimizer.schedule import Noam

class TransformerModelBuilder(BaseModelBuilder):
    def __init__(self, n_features, n_source, n_target, fixe_hparams={}, *args, **kwargs):
        self.n_features = n_features
        self.n_source = n_source
        self.n_target = n_target
        self.fixe_hparams = fixe_hparams
        
        self.loss = kwargs.pop('loss', 'mse')
        self.metrics = kwargs.pop('metrics', 'mae')
        
        self.num_layers_min  = kwargs.pop('num_layers_min', 1)
        self.num_layers_max  = kwargs.pop('num_layers_max', 12)
        self.num_layers_step = kwargs.pop('num_layers_step', 1)
        
        self.d_model_min  = kwargs.pop('d_model_min', 64)
        self.d_model_max  = kwargs.pop('d_model_max', 2048)
        self.d_model_step = kwargs.pop('d_model_step', 64)
        
        self.num_heads_min  = kwargs.pop('num_heads_min', 1)
        self.num_heads_max  = kwargs.pop('num_heads_max', 32)
        self.num_heads_step = kwargs.pop('num_heads_step', 1)
        
        self.dff_min  = kwargs.pop('dff_min', 512)
        self.dff_max  = kwargs.pop('dff_max', 2048)
        self.dff_step = kwargs.pop('dff_step', 128)
        
        self.dropout_rate_min  = kwargs.pop('dropout_rate_min', 0.1)
        self.dropout_rate_max  = kwargs.pop('dropout_rate_max', 0.5)
        self.dropout_rate_step = kwargs.pop('dropout_rate_step', 0.1)
        
        self.warmup_steps_min  = kwargs.pop('warmup_steps_min', 1000)
        self.warmup_steps_max  = kwargs.pop('warmup_steps_max', 10000)
        self.warmup_steps_step = kwargs.pop('warmup_steps_step', 1000)
           
        # Initialize the base class with any remaining arguments.
        super().__init__(*args, **kwargs)
        
    def build_model(self, hp):
        num_layers      = setHyperparameter(hp.Int,   self.fixe_hparams, f'num_layers'  , min_value=self.num_layers_min  , max_value=self.num_layers_max  , step=self.num_layers_step  )
        d_model         = setHyperparameter(hp.Int,   self.fixe_hparams, f'd_model'     , min_value=self.d_model_min     , max_value=self.d_model_max     , step=self.d_model_step     )
        num_heads       = setHyperparameter(hp.Int,   self.fixe_hparams, f'num_heads'   , min_value=self.num_heads_min   , max_value=self.num_heads_max   , step=self.num_heads_step   )
        dff             = setHyperparameter(hp.Int,   self.fixe_hparams, f'dff'         , min_value=self.dff_min         , max_value=self.dff_max         , step=self.dff_step         )
        dropout_rate    = setHyperparameter(hp.Float, self.fixe_hparams, f'dropout_rate', min_value=self.dropout_rate_min, max_value=self.dropout_rate_max, step=self.dropout_rate_step)
        warmup_steps    = setHyperparameter(hp.Int,   self.fixe_hparams, f'warmup_steps', min_value=self.warmup_steps_min, max_value=self.warmup_steps_max, step=self.warmup_steps_step)
        
        transformer = TransformerForecaster(num_layers, d_model, num_heads, dff,
                                            self.n_source, self.n_target, self.n_features, dropout_rate)
        # Use the custom Noam learning rate schedule
        learning_rate = Noam(d_model, warmup_steps=warmup_steps)
        optimizer = tf.keras.optimizers.Adam(learning_rate) # beta_1=0.9, beta_2=0.98, epsilon=1e-9
        
        transformer.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        
        return transformer
    