import tensorflow as tf
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

        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
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

        self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
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

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            "max_target_seq_len": self.max_target_seq_len,
            "dropout_rate": self.dropout.rate,
            "d_model": self.d_model
        })
        return config

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
        self.final_layer = Dense(num_features)

        # Build the model
        dummy_encoder_input = tf.zeros((1, max_input_seq_len, num_features))
        dummy_decoder_input = tf.zeros((1, max_target_seq_len, num_features))
        _ = self((dummy_encoder_input, dummy_decoder_input))

    def create_look_ahead_mask(self, size):
        """
        Creates a lower triangular mask of shape (size, size) to prevent attention being drawn to future tokens.
        """
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def create_combined_mask(self, decoder_mask):
        """
        Combines look-ahead mask with decoder padding mask.
        Args:
            decoder_mask: Shape tensor (batch, T_dec) or (T_dec,) with 1 for valid tokens.
        Returns:
            combined_mask: Form tensor (batch, T_dec, T_dec)
        """
        if len(decoder_mask.shape) == 1:
            decoder_mask = tf.expand_dims(decoder_mask, 0)
        T = tf.shape(decoder_mask)[1]
        look_ahead_mask = self.create_look_ahead_mask(T)
        decoder_mask_cols = tf.expand_dims(decoder_mask, 1)  # (batch, 1, T)
        mask_cols = look_ahead_mask * tf.cast(decoder_mask_cols, look_ahead_mask.dtype)
        decoder_mask_rows = tf.expand_dims(decoder_mask, 2)  # (batch, T, 1)
        combined_mask = mask_cols * tf.cast(decoder_mask_rows, mask_cols.dtype)
        return combined_mask

    def autoregressive_predict(self, encoder_input, start_token, target_seq_len, mask=None):
        """
        Generates a sequence autoregressively given the encoder input.
        
        Args:
            encoder_input: A tensor with dynamic shape. If its rank is 2, we add a batch dimension.
            start_token: A tensor representing the start token (vector of size self.num_features).
            target_seq_len: An integer (or scalar tensor) representing the target sequence length.
            mask: Optional dictionary with keys "encoder" and/or "decoder" providing attention masks.
                The decoder mask is expected to have shape [batch, target_seq_len], with zeros indicating padding.
                
        Returns:
            predictions_stacked: A tensor of shape [batch, target_seq_len, self.num_features] containing the predicted tokens.
        """
        
        # Ensure encoder_input has a batch dimension (dynamic check)
        tf.cond(
            tf.equal(tf.rank(encoder_input), 2),
            lambda: tf.expand_dims(encoder_input, axis=0), 
            lambda: encoder_input
        )
        batch_size = tf.shape(encoder_input)[0]
        
        # Prepare the initial decoder input by tiling the start token over the batch.
        # The start token is reshaped to [1, 1, self.num_features] and tiled to [batch_size, 1, self.num_features].
        decoder_input = tf.tile(tf.reshape(start_token, (1, 1, self.num_features)), [batch_size, 1, 1])
        
        # Create a TensorArray to hold predicted tokens for each generation step.
        predictions_array = tf.TensorArray(dtype=tf.float32, size=target_seq_len)
        
        # Initialize a "finished" flag for each batch element (False means not finished).
        finished = tf.zeros([batch_size], dtype=tf.bool)
        
        # If a decoder mask is provided, extract it.
        decoder_mask = mask["decoder"] if (mask is not None and "decoder" in mask) else None
        # We already use the encoder mask in self((encoder_input, decoder_input, ext_mask), training=False)
        
        # The loop condition: continue until we've generated target_seq_len tokens (excluding the start token).
        def condition(decoder_input, predictions_array, finished):
            # The current number of generated tokens (excluding the initial start token)
            current_generated = tf.shape(decoder_input)[1] - 1
            return current_generated < target_seq_len

        def loop_body(decoder_input, predictions_array, finished):
            current_generated = tf.shape(decoder_input)[1] - 1  # Index of the token to be generated in this iteration
            
            # Build the external mask dictionary for the model call
            ext_mask = {"encoder": mask["encoder"]} if (mask is not None and "encoder" in mask) else None
            
            # Run the model to obtain predictions given the current decoder input.
            preds = self((encoder_input, decoder_input, ext_mask), training=False)
            # The next token is assumed to be the last one in the output sequence
            next_token = preds[:, -1:, :]  # shape: [batch_size, 1, num_features]
            
            # If a decoder mask is provided, check for padded positions.
            if decoder_mask is not None:
                # Get the mask value for the current generation step.
                # Expected shape of decoder_mask: [batch_size, target_seq_len]
                current_mask = decoder_mask[:, current_generated]  # shape: [batch_size]
                # Determine which sequences are finished (mask value equals 0).
                finished_step = tf.equal(current_mask, 0)
                # Update the finished flag for each batch element.
                finished = tf.logical_or(finished, finished_step)
                # For finished sequences, force next_token to zeros.
                # Reshape finished to broadcast over the token shape.
                finished_expanded = tf.reshape(finished, [batch_size, 1, 1])
                next_token = tf.where(finished_expanded, tf.zeros_like(next_token), next_token)
            
            # Write the current prediction into the TensorArray.
            predictions_array = predictions_array.write(current_generated, tf.squeeze(next_token, axis=1))
            
            # Append the next token to the decoder input for the next iteration.
            decoder_input = tf.concat([decoder_input, next_token], axis=1)
            return decoder_input, predictions_array, finished

        # Run the while loop to generate tokens up to target_seq_len.
        decoder_input, predictions_array, finished = tf.while_loop(
            condition,
            loop_body,
            loop_vars=[decoder_input, predictions_array, finished],
            shape_invariants=[
                tf.TensorShape([None, None, self.num_features]),
                tf.TensorShape(None),
                tf.TensorShape([None])
            ]
        )
        
        # Stack the predictions from the TensorArray and transpose to shape [batch, target_seq_len, self.num_features]
        predictions_stacked = predictions_array.stack()  # shape: [target_seq_len, batch, num_features]
        predictions_stacked = tf.transpose(predictions_stacked, perm=[1, 0, 2])
        return predictions_stacked
    
    def __call__(self, inputs, *args, **kwargs):
        # Convert in tensort for accept normal python type in input
        inputs = tf.nest.map_structure(
            lambda x: tf.convert_to_tensor(x)
            if (x is not None and not tf.is_tensor(x) and not isinstance(x, dict))
            else x,
            inputs,
        )
        return super().__call__(inputs, *args, **kwargs)

    def _expand_dims_if(self, tensor, ndims_equal_to, axis):
        if tensor.shape.ndims == ndims_equal_to:
            return tf.expand_dims(tensor, axis=axis)
        return tensor
    
    # batch_size = tf.shape(encoder_input)[0]
    def _mask(self, mask, key, batch_size):
        if mask is not None and mask.get(key, None) is not None:
            mask_key = tf.cast(tf.convert_to_tensor(mask[key]), tf.int32)
            if mask_key.shape.ndims == 1:
                mask_key = tf.tile(tf.expand_dims(mask_key, 0), [batch_size, 1])
        else:
            mask_key = None
        return mask_key

    def call(self, inputs, training=False, mask=None, return_attention=False):
        """
        Main call method supporting multiple input modes.
        
        Supported modes:
        (A) Training mode:
            (encoder_input, decoder_input)
            (encoder_input, decoder_input, mask)
            where:
            - encoder_input: shape (batch, T_enc, n_features) or (T_enc, n_features)
            - decoder_input: shape (batch, T_dec, n_features) or (T_dec, n_features)
            - mask: optional dict with keys "encoder" and/or "decoder"
        
        (B) Autoregressive inference mode:
            (encoder_input, mask)
            (encoder_input, T_enc)
            (encoder_input, T_enc, mask)
            where T_enc can be:
            - a Python int or scalar tensor (same target length for all samples),
            - a 1D tensor of shape (batch,) (per-sample target lengths),
            - a 2D tensor of shape (batch, L) (binary mask indicating valid target positions).
        
        Any 2D encoder_input or decoder_input is automatically expanded to 3D.
        
        If return_attention is True, returns (output, {"encoder_attentions":..., "decoder_attentions":...}).
        """
        # def _convert_input(x):
        #     if x is None or isinstance(x, (int, float, dict)):
        #         return x
        #     if not tf.is_tensor(x):
        #         return tf.convert_to_tensor(x)
        #     return x

        # # Convert each element (except dicts) to a tensor.
        # if isinstance(inputs, (list, tuple)):
        #     inputs = tuple(_convert_input(x) if not isinstance(x, dict) else x for x in inputs)
        # else:
        #     inputs = _convert_input(inputs)
        
        if isinstance(mask, (tuple, list)):
            mask = {
                "encoder": mask[0],
                "decoder": mask[1] if len(mask) > 1 else mask[0]
            }

        encoder_input = None
        decoder_input = None
        target_seq_len = None
        start_token = tf.zeros((self.num_features,), dtype=tf.float32)
        

        if isinstance(inputs, (list, tuple)):
            # if only input sequence
            if len(inputs) == 1:
                inputs = (inputs, self.max_target_seq_len)
                
            if len(inputs) == 2:
                encoder_input, second = inputs

                # If batch size == 1
                encoder_input = self._expand_dims_if(encoder_input, ndims_equal_to=2, axis=0)
                batch_size = tf.shape(encoder_input)[0]
                    
                # case encoder, mask
                if isinstance(second, dict):
                    mask = second
                    
                    # Compute target lengths from the decoder mask.
                    if not 'decoder' in mask:
                        raise AttributeError('If you don\'t provide an output size, you need to provide a decoder mask.')
                    
                    # Get the decoder mask
                    mask = {'encoder':self._mask(mask, 'encoder', batch_size),
                            'decoder':self._mask(mask, 'decoder', batch_size)}
                    
                    # define the target sequence (pad with 0 with mask at 0)
                    target_seq_len = tf.shape(mask['decoder'])[1]

                    # The autoregressive_predict loop will now generate target_seq_len tokens.
                    return self.autoregressive_predict(encoder_input, start_token, target_seq_len, mask=mask)
                
                # case encoder, target_seq_len
                elif isinstance(second, int) or (tf.is_tensor(second) and second.shape.ndims == 0):
                    return self.autoregressive_predict(encoder_input, start_token, target_seq_len=second, mask=mask)
                # case encoder, targets_seq_len
                elif isinstance(second, list) or (tf.is_tensor(second) and second.shape.ndims == 1):
                    target_seq_len = tf.reduce_max(second)  # tsl is the max value in the list
                    
                    # Create a range from 0 to target_seq_len - 1
                    sequence_range = tf.range(target_seq_len)  # Shape: (target_seq_len,)

                    # Expand dimensions to compare against second
                    sequence_range = tf.expand_dims(sequence_range, axis=0)  # Shape: (1, target_seq_len)

                    # Expand `second` (which contains sequence lengths for each batch) to match shape
                    lengths_expanded = tf.expand_dims(second, axis=1)  # Shape: (batch_size, 1)

                    # Create the mask: 1 if sequence index < second[i], else 0
                    mask = tf.cast(sequence_range < lengths_expanded, dtype=tf.float32)  # Shape: (batch_size, target_seq_len)
                    return self.autoregressive_predict(encoder_input, start_token, target_seq_len, mask=mask)
                # case encoder, decoder
                else:
                    decoder_input = self._expand_dims_if(second, ndims_equal_to=2, axis=0)
            elif len(inputs) == 3:
                encoder_input, second, mask = inputs
                
                # If batch size == 1
                encoder_input = self._expand_dims_if(encoder_input, ndims_equal_to=2, axis=0)
                batch_size = tf.shape(encoder_input)[0]
                
                mask = {'encoder':self._mask(mask, 'encoder', batch_size),
                        'decoder':self._mask(mask, 'decoder', batch_size)}
                
                # case encoder, target_seq_len, mask
                if isinstance(second, int) or (tf.is_tensor(second) and second.shape.ndims == 0):
                    return self.autoregressive_predict(encoder_input, start_token, target_seq_len=second, mask=mask)
                # case encoder, targets_seq_len, mask
                elif isinstance(second, list) or (tf.is_tensor(second) and second.shape.ndims == 1):
                    if mask['decoder'] is not None:
                        decoder_mask_tensor = tf.convert_to_tensor(mask["decoder"])
                        target_seq_len = tf.shape(decoder_mask_tensor)[1]
                    else:
                        target_seq_len = tf.reduce_max(tf.cast(second, tf.int32))
                        
                    return  self.autoregressive_predict(encoder_input, start_token, target_seq_len, mask)
                else:
                    decoder_input = self._expand_dims_if(second, ndims_equal_to=2, axis=0)
            else:
                raise ValueError("Unsupported input tuple length.")
        else:
            raise ValueError("Inputs must be provided as a tuple or list.")

        # Get masks
        encoder_mask = mask.get('encoder', None) if mask is not None else None
        decoder_mask = mask.get('decoder', None) if mask is not None else None

        # Build encoder self-attention mask.
        if encoder_mask is not None:
            encoder_mask_expanded = tf.expand_dims(encoder_mask, 1)
            target_seq_len = tf.shape(encoder_input)[1]
            encoder_mask_final = tf.tile(encoder_mask_expanded, [1, target_seq_len, 1])
            encoder_mask_final = tf.cast(encoder_mask_final, tf.bool)
        else:
            encoder_mask_final = None

        # Build look-ahead mask for decoder self-attention.
        target_seq_len = tf.shape(decoder_input)[1]
        look_ahead_mask = self.create_look_ahead_mask(target_seq_len)
        if decoder_mask is not None:
            combined_mask = self.create_combined_mask(decoder_mask)
        else:
            combined_mask = tf.expand_dims(look_ahead_mask, 0)

        # Run encoder
        enc_output = self.encoder(encoder_input, training=training,
                                mask=encoder_mask_final,
                                return_attention=return_attention)
        
        # Run decoder
        dec_output = self.decoder(decoder_input, enc_output, training=training,
                                look_ahead_mask=combined_mask, padding_mask=None,
                                return_attention=return_attention)

        # Prepare attentions infos if desire
        if return_attention:
            enc_output, encoder_attentions = enc_output
            dec_output, decoder_attentions = dec_output
            attentions = {"encoder_attentions": encoder_attentions, "decoder_attentions": decoder_attentions}
        else:
            encoder_attentions, decoder_attentions = None, None

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