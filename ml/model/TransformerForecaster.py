import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, MultiHeadAttention # type: ignore
import numpy as np

""" For display a value in grap mode:
Activate eagerly: tf.config.run_functions_eagerly(True)
convert to numpy: target_seq_len.numpy()
!!!Warning!!! Extremly slow
"""

########################################################################
# Positional Encoding
########################################################################
@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    """
    Implements sinusoidal positional encoding as described in "Attention Is All You Need".
    Precomputes the encoding matrix for a maximum sequence length.
    """
    def __init__(self, max_seq_len, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pos_encoding = self.compute_positional_encoding(max_seq_len, d_model)

    def compute_positional_encoding(self, max_seq_len, d_model):
        pos = np.arange(max_seq_len)[:, np.newaxis]  # (max_seq_len, 1)
        i = np.arange(d_model)[np.newaxis, :]         # (1, d_model)
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates                # (max_seq_len, d_model)
        # Apply sin to even indices and cos to odd indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]      # (1, max_seq_len, d_model)
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        """
        Args:
            inputs: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor with positional encoding added.
        """
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({"max_seq_len": self.max_seq_len, "d_model": self.d_model})
        return config

########################################################################
# Encoder Layer
########################################################################
@tf.keras.utils.register_keras_serializable()
class EncoderLayer(tf.keras.layers.Layer):
    """
    Single layer of the Transformer encoder.
    Consists of:
      - Multi-Head Self-Attention
      - Dropout + Residual Connection + Layer Normalization
      - Feed-Forward Network (FFN) with dropout and residual connection.
      
    Optionally returns attention weights.
    """
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model  // num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training=False, mask=None, return_attention=False):
        if return_attention:
            attn_output, attn_scores = self.mha(x, x, x, attention_mask=mask, return_attention_scores=True)
        else:
            attn_output = self.mha(x, x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        output = self.layernorm2(out1 + ffn_output)
        if return_attention:
            return output, attn_scores
        else:
            return output

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate
        })
        return config

########################################################################
# Encoder
########################################################################
@tf.keras.utils.register_keras_serializable()
class Encoder(tf.keras.layers.Layer):
    """
    Transformer Encoder: projects inputs to d_model dimensions, adds positional encoding,
    applies dropout, and passes data through a stack of EncoderLayer modules.
    """
    def __init__(self, num_layers, d_model, num_heads, dff, max_seq_len, dropout_rate=0.1, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate

        self.input_projection = Dense(d_model)
        self.pos_encoding = PositionalEncoding(max_seq_len, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training=False, mask=None, return_attention=False):
        x = self.input_projection(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        encoder_attentions = []  # List to collect attention weights if requested.
        for layer in self.enc_layers:
            if return_attention:
                x, attn_scores = layer(x, training=training, mask=mask, return_attention=True)
                encoder_attentions.append(attn_scores)
            else:
                x = layer(x, training=training, mask=mask, return_attention=False)
        if return_attention:
            return x, encoder_attentions
        else:
            return x
    
    def compute_output_shape(self, input_shape):
        return (None, None, self.d_model)

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "max_seq_len": self.max_seq_len,
            "dropout_rate": self.dropout_rate
        })
        return config

########################################################################
# Decoder Layer
########################################################################
@tf.keras.utils.register_keras_serializable()
class DecoderLayer(tf.keras.layers.Layer):
    """
    Single layer of the Transformer decoder. Consists of:
      - Self-attention (with look-ahead mask)
      - Encoder-decoder (cross) attention (with padding mask)
      - Feed-Forward Network
    Residual connections, dropout, and layer normalization are applied.
    
    Optionally returns attention weights.
    """
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None, return_attention=False):
        # Self-attention block with look-ahead mask.
        if return_attention:
            attn1_output, attn1_scores = self.mha1(x, x, x, attention_mask=look_ahead_mask, return_attention_scores=True)
        else:
            attn1_output = self.mha1(x, x, x, attention_mask=look_ahead_mask)
        attn1_output = self.dropout1(attn1_output, training=training)
        out1 = self.layernorm1(x + attn1_output)

        # Encoder-decoder (cross) attention block.
        if return_attention:
            attn2_output, attn2_scores = self.mha2(out1, enc_output, enc_output, attention_mask=padding_mask, return_attention_scores=True)
        else:
            attn2_output = self.mha2(out1, enc_output, enc_output, attention_mask=padding_mask)
        attn2_output = self.dropout2(attn2_output, training=training)
        out2 = self.layernorm2(out1 + attn2_output)

        # Feed-Forward Network block.
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        output = self.layernorm3(out2 + ffn_output)

        if return_attention:
            attn_dict = {"self_attention": attn1_scores, "enc_dec_attention": attn2_scores}
            return output, attn_dict
        else:
            return output

    def get_config(self):
        config = super(DecoderLayer, self).get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate
        })
        return config


########################################################################
# Decoder
########################################################################
@tf.keras.utils.register_keras_serializable()
class Decoder(tf.keras.layers.Layer):
    """
    Transformer Decoder: Projects inputs, adds positional encoding, applies dropout,
    and passes data through a stack of DecoderLayer modules.
    
    Supports an optional flag "return_attention" to return attention weights.
    """
    def __init__(self, num_layers, d_model, num_heads, dff, max_target_seq_len, dropout_rate=0.1, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.max_target_seq_len = max_target_seq_len
        self.d_model = d_model
        self.input_projection = Dense(d_model)
        self.pos_encoding = PositionalEncoding(max_target_seq_len, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None, return_attention=False):
        """
        Args:
            x: Decoder input tensor of shape (batch, T_dec, n_features).
            enc_output: Encoder output tensor of shape (batch, T_enc, d_model).
            training: Boolean flag.
            look_ahead_mask: Look-ahead mask for self-attention.
            padding_mask: Optional mask for encoder-decoder attention.
            return_attention: Boolean flag. If True, returns (output, list_of_attention_dicts).
        Returns:
            If return_attention is False: output tensor of shape (batch, T_dec, d_model).
            If True: a tuple (output, attention_list) where attention_list is a list of dictionaries
                    (one per decoder layer) with keys "self_attention" and "enc_dec_attention".
        """
        x = self.input_projection(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        decoder_attentions = [] if return_attention else None
        for layer in self.dec_layers:
            if return_attention:
                x, attn_dict = layer(x, enc_output, training=training,
                                     look_ahead_mask=look_ahead_mask, padding_mask=padding_mask,
                                     return_attention=True)
                decoder_attentions.append(attn_dict)
            else:
                x = layer(x, enc_output, training=training,
                          look_ahead_mask=look_ahead_mask, padding_mask=padding_mask, return_attention=False)
        if return_attention:
            return x, decoder_attentions
        else:
            return x
        
    def compute_output_shape(self, input_shape):
        return (None, None, self.d_model)

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            "max_target_seq_len": self.max_target_seq_len,
            "dropout_rate": self.dropout_rate,
            "d_model": self.d_model
        })
        return config
    
########################################################################
# Final
########################################################################
@tf.keras.utils.register_keras_serializable()
class Final(tf.keras.layers.Dense):
    """

    """
    def compute_output_shape(self, input_shape):
       return (None, None, self.units)

########################################################################
# Transformer Forecaster
########################################################################
@tf.keras.utils.register_keras_serializable()
class TransformerForecaster(tf.keras.Model):
    """
    Transformer Forecaster: Complete encoder-decoder model for time series prediction.
    
    Input modes supported:
      - Training: (encoder_input, decoder_input) or (encoder_input, decoder_input, mask)
      - Autoregressive inference: (encoder_input, target_seq_len)
          where encoder_input can be a 2D (T, n_features) or 3D (batch, T, n_features) tensor.
    
    The model also accepts a “return_attention” Boolean argument in call() to return
    attention weights without first duplicating the propagation code.
    """
    def __init__(self, num_layers, d_model, num_heads, dff,
                 max_input_seq_len, max_target_seq_len, num_features,
                 dropout_rate=0.1, **kwargs):
        super(TransformerForecaster, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.max_input_seq_len = max_input_seq_len
        self.max_target_seq_len = max_target_seq_len
        self.num_features = num_features
        self.dropout_rate = dropout_rate

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, max_input_seq_len, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, max_target_seq_len, dropout_rate)
        self.final_layer = Final(num_features) #Dense(num_features)

        # Build the model
        dummy_encoder_input = tf.zeros((1, max_input_seq_len, num_features))
        dummy_decoder_input = tf.zeros((1, max_target_seq_len, num_features))
        _ = self(encoder_input=dummy_encoder_input, decoder_input=dummy_decoder_input)
    
    def print_layers(self):
        self._print_layers_rec(self, indent=0, visited=None)
        
        
    def _print_layers_rec(self, module, indent=0, visited=None):
        if visited is None:
            visited = set()
        # To avoid infinite loops, we can memorize the ID of objects already visited
        if id(module) in visited:
            return
        visited.add(id(module))
        
        prefix = " " * indent
        # Try to display some useful information
        try:
            params = module.count_params()
        except Exception as e:
            params = "?"
        try:
            name = module.name
        except Exception as e:
            name = module.__class__.__name__
        print(f"{prefix}{name:40s} - {module.__class__.__name__:25s} - Params: {params}")

        # For sub-layers, look at either the "layers" attribute or browse __dict__
        if hasattr(module, "layers"):
            sublayers = module.layers
        else:
            sublayers = []
        # In some cases, lists stored in attributes (e.g. self.enc_layers) are not accessible via module.layers, so you have to iterate through __dict__
        for attr_name, attr_value in vars(module).items():
            if isinstance(attr_value, tf.keras.layers.Layer):
                sublayers.append(attr_value)
            elif isinstance(attr_value, list):
                for item in attr_value:
                    if isinstance(item, tf.keras.layers.Layer):
                        sublayers.append(item)
        # Eliminate duplicates (sometimes the same objects appear in module.layers and via __dict__)
        sublayers = list({id(layer): layer for layer in sublayers}.values())

        # Recursive call for each sub-layer
        for layer in sublayers:
            self._print_layers_rec(layer, indent + 2, visited)
            
    class _DisplayLearningRate(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            opt = self.model.optimizer
            
            # If optimizer implement _decayed_lr (TF ≥2.9) :
            if hasattr(opt, "_decayed_lr"):
                try:
                    lr_t = opt._decayed_lr(tf.float32)
                except Exception:
                    lr_t = opt.learning_rate
            else:
                # Else get learning_rate which can be :
                #    - a tf.Variable
                #    - a LearningRateSchedule
                lr_t = opt.learning_rate

            # If a schedule, cann with the iteration number done
            if isinstance(lr_t, tf.keras.optimizers.schedules.LearningRateSchedule):
                lr_t = lr_t(opt.iterations)

            # convert it to a float
            # If eager mode, convert to numpy
            try:
                lr = lr_t.numpy()
            except Exception:
                lr = tf.keras.backend.get_value(lr_t)

            print(f"Learning rate: {lr:.6e}")
            
    def fit(self, *args, **kwargs):
        # get the callbacks list
        callbacks = list(kwargs.pop('callbacks', []))
        
        callbacks.append(TransformerForecaster._DisplayLearningRate())
        return super().fit(*args, callbacks=callbacks, **kwargs)
        
    def create_look_ahead_mask(self, size):
        """
        Creates a lower triangular mask of shape (size, size) to prevent attention being drawn to future tokens.
        """
        # mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        # return mask
        mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return tf.cast(mask, tf.bool)

    def create_combined_mask(self, decoder_mask):
        """
        Combines look-ahead mask with decoder padding mask.
        Args:
            decoder_mask: Tensor de forme (batch, T_dec) ou (T_dec,) avec 1 pour tokens valides.
        Returns:
            combined_mask: Tensor booléen de forme (batch, T_dec, T_dec)
        """
        # S’assurer qu’on a bien (batch, T)
        if decoder_mask.shape.ndims == 1:
            decoder_mask = tf.expand_dims(decoder_mask, 0)           # (1, T)
        # Taille temporelle
        T = tf.shape(decoder_mask)[1]

        # 1) Look-ahead mask booléen : True pour diag+passé, False pour futur
        look_ahead = self.create_look_ahead_mask(T)                 # (T, T), bool

        # 2) Padding mask du décodeur en booléen
        dec_pad = tf.cast(decoder_mask, tf.bool)                    # (batch, T)

        # 3) On broadcast pour les deux axes q et k
        #    - Axis 1 pour q (requêtes)
        #    - Axis 2 pour k (clé)
        dec_pad_q = tf.expand_dims(dec_pad, 2)                      # (batch, T, 1)
        dec_pad_k = tf.expand_dims(dec_pad, 1)                      # (batch, 1, T)

        # 4) On étend look_ahead sur le batch
        look_ahead = tf.expand_dims(look_ahead, 0)                  # (1, T, T)

        # 5) ET logique entre les 3 masques
        combined_mask = look_ahead & dec_pad_q & dec_pad_k          # (batch, T, T)

        return combined_mask


    def autoregressive_predict(self, encoder_input, target_seq_len, mask=None, return_attention=False):
        batch_size = tf.shape(encoder_input)[0]
        
        # Create a TensorArray for decoder input tokens with fixed size: target_seq_len + 1 (including the start token).
        decoder_input_array = tf.TensorArray(
            dtype=tf.float32, 
            size=target_seq_len + 1, 
            element_shape=tf.TensorShape([None, self.num_features])
        )
        
        dummy = tf.zeros([batch_size, self.num_features], dtype=tf.float32)
        decoder_input_array = decoder_input_array.write(0, dummy)
        
        # Create a TensorArray to hold the predicted tokens.
        predictions_array = tf.TensorArray(
            dtype=tf.float32, 
            size=target_seq_len, 
            element_shape=tf.TensorShape([None, self.num_features])
        )
        
        finished = tf.zeros([batch_size], dtype=tf.bool)
        
        # Loop counter 'i' represents the current length of decoder tokens (starts at 1 because of the start token).
        def condition(i, *_):
            return tf.less(i, target_seq_len + 1)
        
            
        def loop_body(i, decoder_input_array, predictions_array, finished):     
            # Stack all previous tokens (excluding the dummy at index 0) into shape [batch, i-1, num_features]
            prev_tokens = tf.transpose(
                tf.slice(
                    decoder_input_array.stack(),
                    [1, 0, 0],  # skip the start_token slot when stacking for input
                    [i - 1, batch_size, self.num_features]
                ), perm=[1, 0, 2]
            )
            # Add a dummy at the end so the model receives a full-length input. The teacher forcing of the call will remove the last token (dummy one)
            dummy_step = tf.zeros([batch_size, 1, self.num_features], dtype=tf.float32)
            decoder_input = tf.concat([prev_tokens, dummy_step], axis=1)
            
          
            preds = self(
                encoder_input=encoder_input,
                decoder_input=decoder_input,
                mask={"encoder": mask["encoder"]},
                training=False
            )
            next_token = preds[:, -1:, :]  # shape: [batch, 1, num_features]
            
            if mask['decoder'] is not None:
                current_mask = mask['decoder'][:, i - 1]  # shape: [batch]
                finished_step = tf.equal(current_mask, 0)
                finished = tf.logical_or(finished, finished_step)
                finished_expanded = tf.reshape(finished, [batch_size, 1, 1])
                next_token = tf.where(finished_expanded, tf.zeros_like(next_token), next_token)
            
            # Instead of squeezing, explicitly reshape next_token to [batch_size, num_features].
            next_token_reshaped = tf.reshape(next_token, [batch_size, self.num_features])
            predictions_array = predictions_array.write(i - 1, next_token_reshaped)
            decoder_input_array = decoder_input_array.write(i, next_token_reshaped)
            
            return i + 1, decoder_input_array, predictions_array, finished

        i0 = tf.constant(1)
        i, decoder_input_array, predictions_array, finished = tf.while_loop(
            condition,
            loop_body,
            loop_vars=[i0, decoder_input_array, predictions_array, finished],
            shape_invariants=[
                i0.get_shape(),
                tf.TensorShape(None),
                tf.TensorShape(None),
                tf.TensorShape([None])
            ]
        )
        
        predictions_stacked = predictions_array.stack()  # shape: [target_seq_len, batch, num_features]
        predictions_stacked = tf.transpose(predictions_stacked, perm=[1, 0, 2])
        return predictions_stacked

    def _expand_dims_if(self, tensor, ndims_equal_to, axis, error=False):
        if tensor.shape.ndims == ndims_equal_to:
            return tf.expand_dims(tensor, axis=axis)
        elif error:
            raise ValueError(f'Tensor should have {ndims_equal_to} ndims but have {tensor.shape.ndims}.')
        return tensor
    
    def _mask(self, mask, batch_size):
        for key, value in mask.items(): # encoder, decoder
            # ignore if none
            if value is None:
                continue

            mask[key] = tf.cast(tf.convert_to_tensor(value), tf.bool)  # convert to int tensor
            
            # If mask is 1D, expand it to 2D by repeating it for the batch size
            if mask[key].shape.ndims == 1:
                mask[key] = tf.tile(tf.expand_dims(mask[key], 0), [batch_size, 1])
            elif mask[key].shape.ndims != 2:
                raise ValueError(f"Invalid mask shape for {key}: {mask[key].shape}. Expected 1D or 2D tensor.")
            
        return mask
    
    #! return_attention is not yet availible for autoregression
    def call(self, encoder_input, decoder_input=None, target_seq_len=None, mask=None, training=False, return_attention=False):
        """
        Runs a forward pass through the TransformerForecaster model.

        This method supports multiple input configurations in order to flexibly handle
        encoder and decoder inputs, including autoregressive prediction.

        Parameters
        ----------
        encoder_input : tf.Tensor or dict
            - When a tensor: A 3D tensor of shape (batch_size, input_seq_len, num_features)
            representing the encoder input. If a tensor of lower rank is passed (e.g. 2D),
            it is expanded along a new axis at index 0.
            - When a dict: Must contain the key 'encoder_input'. Other keys may include:
            'decoder_input', 'target_seq_len', and 'mask'. These values are popped from the dict
            and used as if passed as separate arguments.

        decoder_input : tf.Tensor, optional
            A 3D tensor of shape (batch_size, target_seq_len, num_features) representing the decoder
            input. If not provided (i.e. None), the model will enter autoregressive prediction mode,
            in which case `target_seq_len` must be provided or derived from the decoder mask.
            
        target_seq_len : int, list, or tf.Tensor, optional
            Specifies the target sequence length for prediction. This is used only when
            `decoder_input` is None (i.e. during autoregressive prediction). When a list or 1D tensor
            is provided, the maximum value is taken to determine the final target sequence length.
            If not provided and no decoder mask is available, `self.max_target_seq_len` is used.

        mask : tuple, list, dict, or None
            Specifies masks for attention operations:
            - If a tuple or list is provided: The first element is used as the encoder mask and
            the second as the decoder mask. If only one element is given, it is used for both.
            - If a dict is provided: Expected keys are 'encoder' and 'decoder'. Missing keys default to None.
            - If None is provided: Both encoder and decoder masks default to None.
            In all cases, the mask is processed to ensure its shape is appropriate for the current batch.

        training : bool, default False
            Indicator for whether the model is running in training mode (affects dropout, etc.).

        return_attention : bool, default False
            If True, the method returns a tuple where the second element is a dictionary containing
            the attention weights from the encoder and decoder layers.

        Returns
        -------
        tf.Tensor or tuple
            - In the typical case, returns the output tensor of the final layer with shape (batch_size, target_seq_len, num_features).
            - If `return_attention` is True, returns a tuple (output, attentions), where "attentions" is a dict with keys:
                * "encoder_attentions": List of attention scores from encoder layers.
                * "decoder_attentions": List of attention scores from decoder layers.

        Behavior
        --------
        - If `encoder_input` is provided as a dict, its keys ('encoder_input', 'decoder_input',
        'target_seq_len', 'mask') are extracted and used.
        - If the encoder input is not 3D, the input is expanded along axis 0.
        - When `decoder_input` is None, autoregressive prediction is invoked using
        `self.autoregressive_predict` with a derived or provided target sequence length.
        - When a decoder mask is not provided but required, the target sequence length is inferred
        from the mask dimensions or defaults to `self.max_target_seq_len`.
        - The decoder input is adjusted by slicing off the last time step and prepending a start token
        derived from the last time step of the encoder input.
        - Appropriate look-ahead masks and encoder masks are generated for attention mechanisms.
        - Finally, the encoder and decoder are run sequentially and the result is passed through the final layer.
        """
        # can't pass more than one input for the training so a dict
        if isinstance(encoder_input, dict):
            encoder_input, decoder_input, target_seq_len, mask = encoder_input.pop('encoder_input'), encoder_input.pop('decoder_input',decoder_input), encoder_input.pop('target_seq_len', target_seq_len), encoder_input.pop('mask', mask)
        
        # Adjust the encoder shape
        if encoder_input.shape.ndims != 3:
            encoder_input = self._expand_dims_if(encoder_input, ndims_equal_to=2, axis=0, error=True)
        
        # get the batch size
        batch_size = tf.shape(encoder_input)[0]
        
        # handle the mask
        if isinstance(mask, (tuple, list)):
            mask = {
                "encoder": mask[0],
                "decoder": mask[1] if len(mask) > 1 else mask[0]
            }
        elif isinstance(mask, dict):
            mask['decoder'] = mask.get('decoder', None)
            mask['encoder'] = mask.get('encoder', None)
        elif mask is None:
            mask = {'encoder': None, 'decoder': None}
        else:
            raise ValueError("Mask should be a tuple, list, dict or None.")
        
        # Make sure of the mask shapes
        mask = self._mask(mask, batch_size)
        
        # If not enough information, set the target sequence len to the max of the training
        if decoder_input is None:
            if target_seq_len is None:
                if mask['decoder'] is None:
                    target_seq_len = self.max_target_seq_len
                else:
                    target_seq_len = tf.shape(mask['decoder'])[1] 
            elif isinstance(target_seq_len, list) or (tf.is_tensor(target_seq_len) and target_seq_len.shape.ndims == 1):
                targets_seq_len = target_seq_len
                target_seq_len = tf.reduce_max(target_seq_len)
                if mask['decoder'] is None:
                    # Create a range from 0 to target_seq_len - 1
                    sequence_range = tf.range(target_seq_len)  # Shape: (target_seq_len,)

                    # Expand dimensions to compare against targets_seq_len
                    sequence_range = tf.expand_dims(sequence_range, axis=0)  # Shape: (1, target_seq_len)

                    # Expand `targets_seq_len` (which contains sequence lengths for each batch) to match shape
                    lengths_expanded = tf.expand_dims(targets_seq_len, axis=1)  # Shape: (batch_size, 1)

                    # Create the mask: 1 if sequence index < targets_seq_len[i], else 0
                    mask['decoder'] = tf.cast(sequence_range < lengths_expanded, dtype=tf.float32)
            elif not isinstance(target_seq_len, int) and not (tf.is_tensor(target_seq_len) and target_seq_len.shape.ndims == 0):
                raise ValueError("target_seq_len should be an integer, a list of integers or None.")
            
            return self.autoregressive_predict(encoder_input, target_seq_len, mask=mask, return_attention=return_attention)
        
        # Adjust the decoder shape    
        if decoder_input.shape.ndims != 3:
            decoder_input = self._expand_dims_if(decoder_input, ndims_equal_to=2, axis=0, error=True)

        #start_token = tf.zeros((self.num_features,), dtype=tf.float32)
        start_token = encoder_input[:, -1:, :]
        
        # Ensure decoder_input is 3D
        decoder_input = self._expand_dims_if(decoder_input, ndims_equal_to=2, axis=0)
        # Remove the last candle from decoder input (along time axis)
        decoder_input_sliced = decoder_input[:, :-1, :]  # (batch, T_dec - 1, n_features)
        # Prepend start token to decoder input
        decoder_input = tf.concat([start_token, decoder_input_sliced], axis=1)

        # Build encoder self-attention mask.
        if mask['encoder'] is not None:
            encoder_mask_expanded = tf.expand_dims(mask['encoder'], 1)
            target_seq_len = tf.shape(encoder_input)[1]
            encoder_mask_final = tf.tile(encoder_mask_expanded, [1, target_seq_len, 1])
            encoder_mask_final = tf.cast(encoder_mask_final, tf.bool)
        else:
            encoder_mask_final = None

        # Build look-ahead mask for decoder self-attention.
        
        if mask['decoder'] is not None:
            combined_mask = self.create_combined_mask(mask['decoder'])
        else:
            target_seq_len = tf.shape(decoder_input)[1]
            combined_mask = tf.expand_dims(self.create_look_ahead_mask(target_seq_len), 0)

        # Run encoder
        enc_output = self.encoder(encoder_input, training=training,
                                mask=encoder_mask_final,
                                return_attention=return_attention)
        
        # Run decoder
        dec_output = self.decoder(decoder_input, enc_output, training=training,
                                look_ahead_mask=combined_mask, padding_mask=encoder_mask_final,
                                return_attention=return_attention)

        # Prepare attentions infos if desire
        if return_attention:
            enc_output, encoder_attentions = enc_output
            dec_output, decoder_attentions = dec_output
            attentions = {"encoder_attentions": encoder_attentions, "decoder_attentions": decoder_attentions}

        # Run final layer
        output = self.final_layer(dec_output)

        return (output, attentions) if return_attention else output

    def get_config(self):
        base_config = super(TransformerForecaster, self).get_config()
        config = {
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "max_input_seq_len": self.max_input_seq_len,
            "max_target_seq_len": self.max_target_seq_len,
            "num_features": self.num_features,
            "dropout_rate": self.dropout_rate,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        if "dtype" in config and isinstance(config["dtype"], dict):
            d_config = config["dtype"]
            if d_config.get("class_name") == "DTypePolicy":
                policy_name = d_config.get("config", {}).get("name", None)
                if policy_name is not None:
                    config["dtype"] = tf.keras.mixed_precision.Policy(policy_name)
        return cls(**config)