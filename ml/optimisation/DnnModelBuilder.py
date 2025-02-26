from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization # type: ignore
from tensorflow.keras.models import Model # type: ignore

from .ModelBuilder import BaseModelBuilder
from ..module import getRegularizers, getOptimizers
from ..layer.module import activation as getActivation, residualConnection, merge as getMerge

class DnnModelBuilder(BaseModelBuilder):
    def __init__(self, n_features, n_labels, *args, **kwargs):
        self.n_features = n_features
        self.n_labels = n_labels
        
        self.min_layers = kwargs.pop('min_layers', 1)
        self.max_layers = kwargs.pop('max_layers', 10)

        self.units_min = kwargs.pop('units_min', 32)
        self.units_max = kwargs.pop('units_max', 2048)
        self.units_step = kwargs.pop('units_step', 32)

        self.dropout_min = kwargs.pop('dropout_min', 0.0)
        self.dropout_max = kwargs.pop('dropout_max', 0.6)
        self.dropout_step = kwargs.pop('dropout_step', 0.1)

        self.negative_slope_min = kwargs.pop('negative_slope_min', 0.001)
        self.negative_slope_max = kwargs.pop('negative_slope_max', 0.5)
        self.negative_slope_step = kwargs.pop('negative_slope_step', 0.001)

        self.regularizer_min = kwargs.pop('regularizer_min', 1e-6)
        self.regularizer_max = kwargs.pop('regularizer_max', 0.01)

        self.merges = kwargs.pop('merges', ['concat','add','average','weighted_avg'])

        self.activations = kwargs.pop('activations', ['relu','elu','selu','gelu','leaky_relu','prelu','mish','softplus'])

        self.optimizers = kwargs.pop('optimizers', ['adam', 'sgd', 'rmsprop'])
        
        super().__init__(*args, **kwargs)
        
    def _choices(self, hp, block_id, i):
        units = hp.Int(f'units_{block_id}_{i}', min_value=self.units_min, max_value=self.units_max,
                               step=self.units_step, default=(self.units_min + self.units_max) // 2)
        activation = hp.Choice(f'activation_{block_id}_{i}', self.activations, default='relu')
        dropout_rate = hp.Float(f'dropout_{block_id}_{i}', self.dropout_min, self.dropout_max,
                                step=self.dropout_step, default=(self.dropout_min + self.dropout_max) / 2)
        # Define negative slope only if using leaky_relu.
        negative_slope = hp.Float(f'negative_slope_{block_id}_{i}', self.negative_slope_min, self.negative_slope_max, step=self.negative_slope_step, 
                                  default=(self.negative_slope_min+self.negative_slope_max)/2, parent_name=f'activation_{block_id}_{i}', parent_values=['leaky_relu'])
        batch_norm = hp.Boolean(f'use_batchnorm_{block_id}_{i}', default=True)
        
        # --- Regularization Options ---
        # Kernel regularizer.
        use_kernel_reg = hp.Boolean(f'use_kernel_regularizer_{block_id}_{i}', default=True)
        with hp.conditional_scope(f'use_kernel_regularizer_{block_id}_{i}', [True]):
            kernel_reg_choice = hp.Choice(f'kernel_regularizer_{block_id}_{i}', ['l1', 'l2', 'l1_l2'], default='l2')
            kernel_reg_factor = hp.Float(f'kernel_regularizer_factor_{block_id}_{i}', self.regularizer_min, self.regularizer_max, default=(self.regularizer_min + self.regularizer_max) / 2)
            kernel_reg_factor2 = hp.Float(f'kernel_regularizer_factor2_{block_id}_{i}', self.regularizer_min, self.regularizer_max, default=(self.regularizer_min + self.regularizer_max) / 2,
                                        parent_name=f'kernel_regularizer_{block_id}_{i}', parent_values=['l1_l2'])
        
        # Bias regularizer.
        use_bias_reg = hp.Boolean(f'use_bias_regularizer_{block_id}_{i}', default=True)
        with hp.conditional_scope(f'use_bias_regularizer_{block_id}_{i}', [True]):
            bias_reg_choice = hp.Choice(f'bias_regularizer_{block_id}_{i}', ['l1', 'l2', 'l1_l2'], default='l2')
            bias_reg_factor = hp.Float(f'bias_regularizer_factor_{block_id}_{i}', self.regularizer_min, self.regularizer_max, default=(self.regularizer_min + self.regularizer_max) / 2)
            bias_reg_factor2 = hp.Float(f'bias_regularizer_factor2_{block_id}_{i}', self.regularizer_min, self.regularizer_max, default=(self.regularizer_min + self.regularizer_max) / 2,
                                        parent_name=f'bias_regularizer_{block_id}_{i}', parent_values=['l1_l2'])
        # Activity regularizer.
        use_activity_reg = hp.Boolean(f'use_activity_regularizer_{block_id}_{i}', default=True)
        with hp.conditional_scope(f'use_activity_regularizer_{block_id}_{i}', [True]):
            activity_reg_choice = hp.Choice(f'activity_regularizer_{block_id}_{i}', 
                                            ['l1', 'l2', 'l1_l2'], default='l2')
            activity_reg_factor = hp.Float(f'activity_regularizer_factor_{block_id}_{i}', self.regularizer_min, self.regularizer_max, default=(self.regularizer_min + self.regularizer_max) / 2)
            activity_reg_factor2 = hp.Float(f'activity_regularizer_factor2_{block_id}_{i}', self.regularizer_min, self.regularizer_max, default=(self.regularizer_min + self.regularizer_max) / 2,
                                        parent_name=f'activity_regularizer_{block_id}_{i}', parent_values=['l1_l2'])
        
        # if need two value stay else take the one value
        kernel_reg_factor2 = kernel_reg_factor2 if kernel_reg_choice == 'l1_l2' else kernel_reg_factor
        bias_reg_factor2 = bias_reg_factor2 if bias_reg_choice == 'l1_l2' else bias_reg_factor
        activity_reg_factor2 = activity_reg_factor2 if activity_reg_choice == 'l1_l2' else activity_reg_factor
        
        # Build the regularizers if enabled.
        kernel_reg = getRegularizers(kernel_reg_choice, kernel_reg_factor, kernel_reg_factor2) if use_kernel_reg else None
        bias_reg = getRegularizers(bias_reg_choice, bias_reg_factor, bias_reg_factor2) if use_bias_reg else None
        activity_reg = getRegularizers(activity_reg_choice, activity_reg_factor, activity_reg_factor2) if use_activity_reg else None
        
        return {'units':units, 'activation':activation, 'dropout_rate':dropout_rate, 'negative_slope':negative_slope, 'batch_norm':batch_norm, 'use_kernel_reg':use_kernel_reg, 'kernel_reg_choice':kernel_reg_choice, 'kernel_reg_factor':kernel_reg_factor, 'use_bias_reg':use_bias_reg, 'bias_reg_choice':bias_reg_choice, 'bias_reg_factor':bias_reg_factor, 'use_activity_reg':use_activity_reg, 'activity_reg_choice':activity_reg_choice, 'activity_reg_factor':activity_reg_factor, 'kernel_reg':kernel_reg, 'bias_reg':bias_reg, 'activity_reg':activity_reg}
        
    def completeDenseLayer(self, x, units, activation, dropout_rate, batch_norm, kernel_reg, bias_reg, activity_reg):
        x = Dense(units, activation=None, kernel_regularizer=kernel_reg, bias_regularizer=bias_reg, activity_regularizer=activity_reg)(x)
        x = getActivation(activation)(x)
            
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
            
        if batch_norm:
            x = BatchNormalization()(x)    
        return x
    
    def block(self, hp, x, block_id):
        """
        Build a dense block with a tunable number of layers.
        
        For each layer index in 0...self.max_layers-1, we use hp.conditional_scope so that
        the hyperparameters (units, activation, dropout, etc.) are only registered if the
        current layer index is within the active number of layers.
        
        Parameters:
            x: Input tensor.
            block_id: Identifier string for this block (used for hyperparameter naming).
            
        Returns:
            The output tensor after applying the dense block.
        """
        x_in = x  # For potential skip connection.
        
        # Option to use a residual connection for this block.
        use_skip = hp.Boolean(f'use_skip_{block_id}', default=True)
        
        # Choose how many layers to activate in this block
        n_layers = hp.Int(f'n_layers_{block_id}', self.min_layers, self.max_layers, default=3) or 0
        for i in range(self.max_layers):
            # This scope is active only if the chosen n_layers is >= i+1.
            with hp.conditional_scope(f'n_layers_{block_id}', list(range(i+1, self.max_layers+1))):
                choice = self._choices(hp, block_id, i)
                
            # Only add the layer if it is within the active n_layers.
            if i < n_layers:
                x = self.completeDenseLayer(x, choice["units"], choice["activation"], choice["dropout_rate"], choice["batch_norm"], choice["kernel_reg"], choice["bias_reg"], choice["activity_reg"])
                    
        # Apply a residual connection if enabled.
        if use_skip:
            x = residualConnection(x_in, x)

        return x

    def build_model(self, hp):
 
        # Input layer
        inputs = Input(shape=(self.n_features,), name='input')
        x = inputs

        # Optional separate start branch
        use_start_branch = hp.Boolean('use_start_branch', default=True)
        with hp.conditional_scope(f'use_start_branch', [True]):
            
            merge_mode_start = hp.Choice('merge_mode_start', self.merges, default=self.merges[0])
            if merge_mode_start not in ('concat',  None):
                self.min_layers -= 1
                self.min_layers -= 1
                with hp.conditional_scope(f'merge_mode_start', [item for item in self.merges if item != 'concat']):
                    choice = self._choices(hp, 'start_merge', '')                
                    start_branches = [self.completeDenseLayer(self.block(hp, x, block_id=f'start_{i}'), choice["units"], choice["activation"], choice["dropout_rate"], choice["batch_norm"], choice["kernel_reg"], choice["bias_reg"], choice["activity_reg"]) for i in range(self.n_labels)]
                self.min_layers += 1
                self.min_layers += 1
            elif merge_mode_start is not None:
                start_branches = [self.block(hp, x, block_id=f'start_{i}') for i in range(self.n_labels)]
                
            if merge_mode_start is not None:
                x = getMerge(merge_mode_start, name='start')(start_branches)
        # Optional main sequential dense block
        use_middle_block = hp.Boolean('use_middle_block', default=True)
        with hp.conditional_scope(f'use_middle_block', [True]):
            x = self.block(hp, x, block_id='middle')

        # Optional end up by separate branchs
        use_end_branch = hp.Boolean('use_end_branch', default=True)
        with hp.conditional_scope(f'use_end_branch', [True]):
            # For each labels, create a separate branch for its prediction
            outputs = Concatenate(name='output')([Dense(1, activation='linear', name=f'output_{i}')(self.block(hp, x, block_id=f'end_{i}'))  for i in range(self.n_labels)])
        with hp.conditional_scope(f'use_end_branch', [False]):
            outputs = Dense(self.n_labels, activation='linear', name='output')(x)

        model = Model(inputs=inputs, outputs=outputs, name='tunable_model')

        # Optimizer selection using conditional scopes.
        optimizer_choice = hp.Choice('optimizer', self.optimizers, default=self.optimizers[0])
        
        lr = hp.Float(f'{optimizer_choice}_lr', min_value=self.learning_rate_min, max_value=self.learning_rate_max, sampling='log', default=1e-3)
        
        model.compile(optimizer=getOptimizers(optimizer_choice, lr), loss='mse', metrics=['mae'])
        
        return model
    
"""Usage

# Binding du constructeur de modèle avec les dimensions des données
dnn_model_builder = ml.optimisation.DnnModelBuilder(n_features=train_data.shape[-1], n_labels=train_labels.shape[-1])
model_builder = dnn_model_builder.build_model

# Initialisation du tuner Hyperband
tuner = ml.optimisation.BayesianTuner(
    model_builder,
    objective='val_loss',      
    max_trials=10000, 
    directory=str(result_path),  # Dossier pour sauvegarder les résultats
    project_name=id_       # Nom du projet
)

# Lancer la recherche d'hyperparamètres
tuner.search(
    train_data, train_labels,
    validation_data=(val_data, val_labels),
    epochs=100,
    verbose=1
)

# Récupération des meilleurs hyperparamètres trouvés
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

# Reconstruction et entraînement du modèle avec les meilleurs hyperparamètres
best_model = tuner.hypermodel.build(best_hp)

# Sauvegarde du meilleur modèle entraîné
best_model.save(result_path / f'best_model_{id_}.keras')

print("Meilleurs hyperparamètres trouvés :")
for key, value in best_hp.values.items():
    print(f'{key} : {value}')
"""