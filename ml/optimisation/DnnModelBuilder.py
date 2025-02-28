from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization # type: ignore
from tensorflow.keras.models import Model # type: ignore

from .ModelBuilder import BaseModelBuilder
from ..module import getRegularizers, getOptimizers
from ..layer.module import activation as getActivation, residualConnection, merge as getMerge

class DnnModelBuilder(BaseModelBuilder):
    def __init__(self, n_features, n_labels, *args, **kwargs):
        """
        Initialize the DNN model builder with hyperparameter configurations.

        Parameters:
            n_features (int): Number of input features.
            n_labels (int): Number of output labels.

        Optional Keyword Arguments:
            # Layer architecture settings
            min_layers (int): Minimum number of layers. Default is 1.
            max_layers (int): Maximum number of layers. Default is 10.
            units_min (int): Minimum number of units per layer. Default is 32.
            units_max (int): Maximum number of units per layer. Default is 2048.
            units_step (int): Step increment for units per layer. Default is 32.

            # Batch normalization settings
            batch_norm (bool): Whether to use batch normalization. Default is True.
            batch_norm_obligate (bool): Whether batch normalization is mandatory. Default is False.

            # Dropout configuration
            dropout_min (float): Minimum dropout rate. Default is 0.0.
            dropout_max (float): Maximum dropout rate. Default is 0.6.
            dropout_step (float): Step increment for dropout rate. Default is 0.1.

            # Activation function (leaky variants) negative slope settings
            negative_slope_min (float): Minimum negative slope. Default is 0.001.
            negative_slope_max (float): Maximum negative slope. Default is 0.5.
            negative_slope_step (float): Step increment for negative slope. Default is 0.001.

            # Regularization settings
            regularizer (bool): Enable regularization. Default is True.
            regularizer_kernel (bool): Regularize kernel weights. Default is True.
            regularizer_bias (bool): Regularize bias terms. Default is True.
            regularizer_activity (bool): Regularize layer activations. Default is True.
            regularizer_min (float): Minimum regularization factor. Default is 1e-6.
            regularizer_max (float): Maximum regularization factor. Default is 0.01.

            # Other hyperparameter options
            merges (list): List of merge operations. Default is ['concat', 'add', 'average', 'weighted_avg'].
            activations (list): List of activation functions to consider.
                Default is ['relu', 'elu', 'selu', 'gelu', 'leaky_relu', 'prelu', 'mish', 'softplus'].
            optimizers (list): List of optimizers to consider. Default is ['adam', 'sgd', 'rmsprop'].
        """
        # Input/Output dimensions
        self.n_features = n_features
        self.n_labels = n_labels
        
        # ---- Layer Architecture ----
        self.min_layers = kwargs.pop('min_layers', 1)
        self.max_layers = kwargs.pop('max_layers', 10)
        self.units_min = kwargs.pop('units_min', 32)
        self.units_max = kwargs.pop('units_max', 2048)
        self.units_step = kwargs.pop('units_step', 32)

        # ---- Batch Normalization Settings ----
        self.batch_norm = kwargs.pop('batch_norm', True)
        self.batch_norm_obligate = kwargs.pop('batch_norm_obligate', False)
        
        # ---- Dropout Settings ----
        self.dropout_min = kwargs.pop('dropout_min', 0.0)
        self.dropout_max = kwargs.pop('dropout_max', 0.6)
        self.dropout_step = kwargs.pop('dropout_step', 0.1)

         # ---- Negative Slope for Leaky Activations ----
        self.negative_slope_min = kwargs.pop('negative_slope_min', 0.001)
        self.negative_slope_max = kwargs.pop('negative_slope_max', 0.5)
        self.negative_slope_step = kwargs.pop('negative_slope_step', 0.001)

        # ---- Regularization Settings ----
        self.regularizer            = kwargs.pop('regularizer', True)
        self.regularizer_kernel     = kwargs.pop('regularizer_kernel', True)
        self.regularizer_bias       = kwargs.pop('regularizer_bias', True)
        self.regularizer_activity   = kwargs.pop('regularizer_activity', True)
        self.regularizer_min        = kwargs.pop('regularizer_min', 1e-6)
        self.regularizer_max        = kwargs.pop('regularizer_max', 0.01)
        
         # If regularization is disabled, disable all specific regularizers
        if self.regularizer is False:
            self.regularizer_kernel, self.regularizer_bias, self.regularizer_activity = False, False, False
        
        # ---- Additional Hyperparameters ----
        self.merges = kwargs.pop('merges', ['concat','add','average','weighted_avg'])
        self.activations = kwargs.pop('activations', ['relu','elu','selu','gelu','leaky_relu','prelu','mish','softplus'])
        self.optimizers = kwargs.pop('optimizers', ['adam', 'sgd', 'rmsprop'])
        
        # Initialize the base class with any remaining arguments.
        super().__init__(*args, **kwargs)
        
    def _regularizer(self, hp, block_id, i, is_desire, name):
        """
        Configure a regularizer for a given hyperparameter tuning block if desired.

        Parameters:
            hp: Hyperparameter tuning object.
            block_id (str): Identifier for the block.
            i (int): Index of the current layer.
            is_desire (bool): Flag indicating whether to apply this regularizer.
            name (str): Name prefix for the regularizer type (e.g., 'kernel', 'bias', 'activity').

        Returns:
            A configured regularizer object (via getRegularizers) or None if not desired.
        """
        if is_desire:
            # Determine if the regularizer should be applied.
            use_reg = hp.Boolean(f'use_{name}_regularizer_{block_id}_{i}', default=True)
            with hp.conditional_scope(f'use_{name}_regularizer_{block_id}_{i}', [True]):
                # Choose the regularizer type: 'l1', 'l2', or 'l1_l2'
                reg_choice = hp.Choice(f'{name}_regularizer_{block_id}_{i}', ['l1', 'l2', 'l1_l2'], default='l2')
                
                # Primary regularization factor.
                reg_factor = hp.Float(f'{name}_regularizer_factor_{block_id}_{i}', self.regularizer_min, self.regularizer_max, default=(self.regularizer_min + self.regularizer_max) / 2)
                
                # Secondary factor needed only when using 'l1_l2'
                reg_factor2 = hp.Float(f'{name}_regularizer_factor2_{block_id}_{i}', self.regularizer_min, self.regularizer_max, default=(self.regularizer_min + self.regularizer_max) / 2,
                                            parent_name=f'{name}_regularizer_{block_id}_{i}', parent_values=['l1_l2'])
            # If l1_l2 is not chosen, use the primary factor for both.
            reg_factor2 = reg_factor2 if reg_choice == 'l1_l2' else reg_factor
            return getRegularizers(reg_choice, reg_factor, reg_factor2) if use_reg else None
        else:
            return None
        
    def _choices(self, hp, block_id, i):
        """
        Define a dictionary of hyperparameter choices for a given layer.

        Parameters:
            hp: Hyperparameter tuning object.
            block_id (str): Identifier for the block.
            i (int): Index of the layer within the block.

        Returns:
            A dictionary containing:
                - units: Number of neurons.
                - activation: Activation function.
                - dropout_rate: Dropout rate.
                - negative_slope: Negative slope for leaky activations (if applicable).
                - batch_norm: Whether to use batch normalization.
                - kernel_reg: Kernel regularizer.
                - bias_reg: Bias regularizer.
                - activity_reg: Activity regularizer.
        """
        units = hp.Int(f'units_{block_id}_{i}', min_value=self.units_min, max_value=self.units_max,
                               step=self.units_step, default=(self.units_min + self.units_max) // 2)
        
        activation = hp.Choice(f'activation_{block_id}_{i}', self.activations, default='relu')
        
        
        dropout_rate = hp.Float(f'dropout_{block_id}_{i}', self.dropout_min, self.dropout_max,
                                step=self.dropout_step, default=(self.dropout_min + self.dropout_max) / 2)
        
        # Define negative slope only if using leaky_relu.
        negative_slope = hp.Float(f'negative_slope_{block_id}_{i}', self.negative_slope_min, self.negative_slope_max, step=self.negative_slope_step, 
                                  default=(self.negative_slope_min+self.negative_slope_max)/2, parent_name=f'activation_{block_id}_{i}', parent_values=['leaky_relu'])
        
        
        batch_norm = hp.Boolean(f'use_batchnorm_{block_id}_{i}', default=True) if self.batch_norm else None
        
        # --- Regularization Options ---
        kernel_reg      = self._regularizer(hp, block_id, i, is_desire=self.regularizer_kernel    , name='kernel')
        bias_reg        = self._regularizer(hp, block_id, i, is_desire=self.regularizer_bias      , name='bias')
        activity_reg    = self._regularizer(hp, block_id, i, is_desire=self.regularizer_activity  , name='activity')
                
        return {
            'units':units, 
            'activation':activation, 
            'dropout_rate':dropout_rate,
            'negative_slope':negative_slope, 
            'batch_norm':batch_norm, 
            'kernel_reg':kernel_reg, 
            'bias_reg':bias_reg, 
            'activity_reg':activity_reg}
        
    def completeDenseLayer(self, x, units, activation, dropout_rate, batch_norm, kernel_reg, bias_reg, activity_reg):
        """
        Build a dense layer with configurable activation, dropout, and batch normalization.

        Parameters:
            x: Input tensor.
            units (int): Number of neurons in the layer.
            activation (str): Activation function.
            dropout_rate (float): Dropout rate to apply after activation.
            batch_norm (bool): Whether to apply batch normalization.
            kernel_reg: Regularizer for kernel weights.
            bias_reg: Regularizer for bias.
            activity_reg: Regularizer for activations.

        Returns:
            Output tensor after applying the Dense layer, activation, optional dropout, and batch normalization.
        """
        x = Dense(units, activation=None, kernel_regularizer=kernel_reg, bias_regularizer=bias_reg, activity_regularizer=activity_reg)(x)
        x = getActivation(activation)(x)
            
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        
        # Apply batch normalization if enabled either by global settings or per-layer choice.
        if (self.batch_norm and batch_norm) or self.batch_norm_obligate:
            x = BatchNormalization()(x)    
        return x
    
    def block(self, hp, x, block_id, min_layers=None, max_layers=None):
        """
        Build a dense block comprising multiple layers with tunable hyperparameters.

        Hyperparameters for each layer are registered conditionally based on the chosen
        number of active layers. Optionally, a residual (skip) connection is added.

        Parameters:
            hp: Hyperparameter tuning object.
            x: Input tensor.
            block_id (str): Unique identifier for the block.
            min_layers (int, optional): Minimum layers to include (defaults to self.min_layers).
            max_layers (int, optional): Maximum layers to include (defaults to self.max_layers).

        Returns:
            Output tensor after applying the dense block (including a residual connection if enabled).
        """
        # Determine effective layer limits.
        max_layers_use = self.max_layers if max_layers is None else max_layers
        min_layers_use = self.min_layers if min_layers is None else min_layers
        
        x_in = x  # For potential skip connection.
        
        # Option to use a residual connection for this block.
        use_skip = hp.Boolean(f'use_skip_{block_id}' , default=True) 
        
        # Choose how many layers to activate in this block
        n_layers = hp.Int(f'n_layers_{block_id}', min_layers_use, max_layers_use, default=3)
        for i in range(max_layers_use):
            # Only register hyperparameters if layer index i is within the active number.
            with hp.conditional_scope(f'n_layers_{block_id}', list(range(i+1, max_layers_use+1))):
                choice = self._choices(hp, block_id, i)
                
                # Only add the layer if it is within the active n_layers.
                if i < n_layers:
                    x = self.completeDenseLayer(x, choice["units"], choice["activation"], choice["dropout_rate"], choice["batch_norm"], choice["kernel_reg"], choice["bias_reg"], choice["activity_reg"])
                    
        # Apply a residual connection if enabled.
        if use_skip:
            x = residualConnection(x_in, x)

        return x

    def build_model(self, hp):
        """
        Construct and compile the tunable model using hyperparameter selections.

        The model can optionally include a start branch (with separate branches per label),
        a middle dense block, and an end branch. The optimizer is also selected and tuned.

        Parameters:
            hp: Hyperparameter tuning object.

        Returns:
            A compiled Keras model.
        """
        # Exclude 'concat' from alternative merge modes.
        merges_without_concat = [item for item in self.merges if item != 'concat']
        
        # Input layer
        inputs = Input(shape=(self.n_features,), name='input')
        x = inputs

        # Optional separate start branch
        use_start_branch = hp.Boolean('use_start_branch', default=True)
        with hp.conditional_scope(f'use_start_branch', [True]):
            
            merge_mode_start = hp.Choice('merge_mode_start', self.merges, default=merges_without_concat[0] if len(merges_without_concat) > 0 else self.merges[0])
            if merge_mode_start is not None:
                max_layers_use = self.max_layers
                min_layers_use = self.min_layers

                if merge_mode_start in merges_without_concat:
                    max_layers_use -= 1
                    min_layers_use -=1
                    
                start_branches = [self.block(hp, x, block_id=f'start_{i}', min_layers=min_layers_use, max_layers=max_layers_use) for i in range(self.n_labels)]
                
                with hp.conditional_scope(f'merge_mode_start', merges_without_concat):
                    choice = self._choices(hp, 'start_merge', '')
                    if merge_mode_start in merges_without_concat:
                        start_branches = [self.completeDenseLayer(block, choice["units"], choice["activation"], choice["dropout_rate"], choice["batch_norm"], choice["kernel_reg"], choice["bias_reg"], choice["activity_reg"]) for block in start_branches]
                
                x = getMerge(merge_mode_start, name='start')(start_branches)
                
        # Optional main sequential dense block
        use_middle_block = hp.Boolean('use_middle_block', default=True)
        with hp.conditional_scope(f'use_middle_block', [True]):
            if use_middle_block:
                x = self.block(hp, x, block_id='middle')

        # Optional end up by separate branchs
        use_end_branch = hp.Boolean('use_end_branch', default=True)
        with hp.conditional_scope(f'use_end_branch', [True]):
            # For each labels, create a separate branch for its prediction
            if use_end_branch:
                outputs = Concatenate(name='output')([Dense(1, activation='linear', name=f'output_{i}')(self.block(hp, x, block_id=f'end_{i}'))  for i in range(self.n_labels)])
        with hp.conditional_scope(f'use_end_branch', [False]):
            if not use_end_branch:
                outputs = Dense(self.n_labels, activation='linear', name='output')(x)

        model = Model(inputs=inputs, outputs=outputs, name='tunable_model')

        # Optimizer selection using conditional scopes.
        optimizer_choice = hp.Choice('optimizer', self.optimizers, default=self.optimizers[0])
        
        lrs = []
        for optimizer in self.optimizers:
            with hp.conditional_scope('optimizer', [optimizer]):
                lrs.append(hp.Float(f'{optimizer}_lr', min_value=self.learning_rate_min, max_value=self.learning_rate_max, sampling='log', default=1e-3))

        lr = lrs[self.optimizers.index(optimizer_choice)]
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