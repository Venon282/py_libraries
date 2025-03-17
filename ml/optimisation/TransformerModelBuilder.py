from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization # type: ignore
from tensorflow.keras.models import Model # type: ignore
import tensorflow as tf

from .ModelBuilder import BaseModelBuilder
from ..module import getRegularizers, getOptimizers
from ..layer.module import activation as getActivation, residualConnection, merge as getMerge
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
        self.metrics = kwargs.pop('loss', 'mae')
        
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
           
        # Initialize the base class with any remaining arguments.
        super().__init__(*args, **kwargs)
        
    def build_model(self, hp):
        num_layers      = setHyperparameter(hp.Int,   self.fixe_hparams, f'num_layers'  , min_value=self.num_layers_min  , max_value=self.num_layers_max  , step=self.num_layers_step  )
        d_model         = setHyperparameter(hp.Int,   self.fixe_hparams, f'd_model'     , min_value=self.d_model_min     , max_value=self.d_model_max     , step=self.d_model_step     )
        num_heads       = setHyperparameter(hp.Int,   self.fixe_hparams, f'num_heads'   , min_value=self.num_heads_min   , max_value=self.num_heads_max   , step=self.num_heads_step   )
        dff             = setHyperparameter(hp.Int,   self.fixe_hparams, f'dff'         , min_value=self.dff_min         , max_value=self.dff_max         , step=self.dff_step         )
        dropout_rate    = setHyperparameter(hp.Float, self.fixe_hparams, f'dropout_rate', min_value=self.dropout_rate_min, max_value=self.dropout_rate_max, step=self.dropout_rate_step)
        
        transformer = TransformerForecaster(num_layers, d_model, num_heads, dff,
                                            self.n_source, self.n_target, self.n_features, dropout_rate)
        # Use the custom Noam learning rate schedule
        learning_rate = Noam(d_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate) # beta_1=0.9, beta_2=0.98, epsilon=1e-9
        
        transformer.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)

        return transformer
    
# """Usage

# # Binding du constructeur de modèle avec les dimensions des données
# dnn_model_builder = ml.optimisation.DnnModelBuilder(n_features=train_data.shape[-1], 
#                                                     n_labels_linear=2,
#                                                     n_labels_sigmoid=0,
#                                                     labels_softmax=[4, 2 , 1]]) # multiclassification. Numbers are the number of class for each
# model_builder = dnn_model_builder.build_model

# # Initialisation du tuner Hyperband
# tuner = ml.optimisation.BayesianTuner(
#     model_builder,
#     objective='val_loss',      
#     max_trials=10000, 
#     directory=str(result_path),  # Dossier pour sauvegarder les résultats
#     project_name=id_       # Nom du projet
# )

# # Lancer la recherche d'hyperparamètres
# tuner.search(
#     train_data, train_labels,
#     validation_data=(val_data, val_labels),
#     epochs=100,
#     verbose=1
# )

# # Récupération des meilleurs hyperparamètres trouvés
# best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

# # Reconstruction et entraînement du modèle avec les meilleurs hyperparamètres
# best_model = tuner.hypermodel.build(best_hp)

# # Sauvegarde du meilleur modèle entraîné
# best_model.save(result_path / f'best_model_{id_}.keras')

# print("Meilleurs hyperparamètres trouvés :")
# for key, value in best_hp.values.items():
#     print(f'{key} : {value}')
# """