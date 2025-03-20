import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, MultiHeadAttention
import numpy as np

# ----------------------------
# Positional Encoding
# ----------------------------

@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_encoding = self.compute_positional_encoding(seq_len, d_model)

    def compute_positional_encoding(self, seq_len, d_model):
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "seq_len": self.seq_len,
            "d_model": self.d_model
        })
        return config
    
# ----------------------------
# Encoder Layer
# ----------------------------
@tf.keras.utils.register_keras_serializable()
class EncoderLayer(tf.keras.layers.Layer):
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

    def call(self, x, training=False, mask=None):
        attn_output = self.mha(x, x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate
        })
        return config


# ----------------------------
# Encoder (Projects continuous inputs via Dense)
# ----------------------------
@tf.keras.utils.register_keras_serializable()
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_seq_len, dropout_rate=0.1, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.input_seq_len = input_seq_len
        self.dropout_rate = dropout_rate

        # Project continuous candlestick data into a d_model–dimensional space.
        self.input_projection = Dense(d_model)
        self.pos_encoding = PositionalEncoding(input_seq_len, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        x = self.input_projection(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for layer in self.enc_layers:
            x = layer(x, training=training, mask=mask)
        return x

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "input_seq_len": self.input_seq_len,
            "dropout_rate": self.dropout_rate,
            # Assuming all EncoderLayers share same num_heads and dff
            "num_heads": self.enc_layers[0].num_heads if self.enc_layers else None,
            "dff": self.enc_layers[0].ffn.layers[0].units if self.enc_layers else None,
        })
        return config

# ----------------------------
# Decoder Layer
# ----------------------------
@tf.keras.utils.register_keras_serializable()
class DecoderLayer(tf.keras.layers.Layer):
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

    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        # Self-attention with look-ahead mask (for autoregressive decoding)
        attn1 = self.mha1(x, x, x, attention_mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)
        # Encoder-decoder attention
        attn2 = self.mha2(out1, enc_output, enc_output, attention_mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        # Feed-forward network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)

    def get_config(self):
        config = super(DecoderLayer, self).get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate
        })
        return config

# ----------------------------
# Decoder (For continuous outputs)
# ----------------------------
@tf.keras.utils.register_keras_serializable()
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_seq_len, dropout_rate=0.1, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.target_seq_len = target_seq_len
        self.dropout_rate = dropout_rate

        # Project target candlestick features into d_model dimensions.
        self.input_projection = Dense(d_model)
        self.pos_encoding = PositionalEncoding(target_seq_len, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        x = self.input_projection(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for layer in self.dec_layers:
            x = layer(x, enc_output, training=training,
                      look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
        return x

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "target_seq_len": self.target_seq_len,
            "dropout_rate": self.dropout_rate,
            "num_heads": self.dec_layers[0].num_heads if self.dec_layers else None,
            "dff": self.dec_layers[0].ffn.layers[0].units if self.dec_layers else None,
        })
        return config

# ----------------------------
# Transformer Forecaster (Encoder-Decoder Model)
# ----------------------------


# Assuming Encoder and Decoder are defined similarly in your model
# For example:
# from your_module import Encoder, Decoder
@tf.keras.utils.register_keras_serializable()
class TransformerForecaster(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_seq_len, target_seq_len, num_features,
                 dropout_rate=0.1, **kwargs):
        # Pass extra keyword arguments to the parent class
        super(TransformerForecaster, self).__init__(**kwargs)

        # Save initialization parameters for serialization
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.input_seq_len = input_seq_len
        self.target_seq_len = target_seq_len
        self.num_features = num_features
        self.dropout_rate = dropout_rate

        # Build the model components (Encoder and Decoder must be defined accordingly)
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_seq_len, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_seq_len, dropout_rate)
        self.final_layer = Dense(num_features)
        
        # build the model
        self.build([(None, self.input_seq_len, self.num_features), 
                    (None, self.target_seq_len, self.num_features)])
    def build(self, input_shape):
        """
        Build the model components using the provided input shapes.
        The expected input_shape is a tuple:
          (encoder_input_shape, decoder_input_shape)
        where each is a tuple of (batch_size, sequence_length, num_features).
        """
        encoder_input_shape, decoder_input_shape = input_shape

        # Build the encoder. Its input shape is (None, input_seq_len, num_features).
        self.encoder.build(encoder_input_shape)

        # The encoder outputs a tensor of shape (None, input_seq_len, d_model).
        encoder_output_shape = (encoder_input_shape[0], self.input_seq_len, self.d_model)

        # Build the decoder.
        # The decoder takes two inputs: one of shape (None, target_seq_len, num_features)
        # and one of shape (None, input_seq_len, d_model).
        self.decoder.build([decoder_input_shape, encoder_output_shape])

        # Build the final Dense layer. Its input shape is (None, target_seq_len, d_model).
        final_layer_input_shape = (decoder_input_shape[0], decoder_input_shape[1], self.d_model)
        self.final_layer.build(final_layer_input_shape)

        # Mark the model as built.
        super(TransformerForecaster, self).build(input_shape)

    def call(self, inputs, training=False):
        # Expecting inputs as a tuple: (encoder_input, decoder_input)
        enc_input, dec_input = inputs
        enc_output = self.encoder(enc_input, training=training)
        # Create look-ahead mask based on decoder sequence length
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(dec_input)[1])
        dec_output = self.decoder(dec_input, enc_output, training=training, look_ahead_mask=look_ahead_mask)
        final_output = self.final_layer(dec_output)
        return final_output

    def create_look_ahead_mask(self, size):
        # Create a lower triangular matrix mask for causal decoding
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def get_config(self):
        # Retrieve the base configuration from the parent class
        base_config = super(TransformerForecaster, self).get_config()
        # Custom configuration for our parameters
        config = {
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "input_seq_len": self.input_seq_len,
            "target_seq_len": self.target_seq_len,
            "num_features": self.num_features,
            "dropout_rate": self.dropout_rate,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        # if "dtype" in config and isinstance(config["dtype"], str):
        #    config["dtype"] = tf.keras.mixed_precision.Policy(config["dtype"])
        if "dtype" in config and isinstance(config["dtype"], dict):
            d_config = config["dtype"]
            if d_config.get("class_name") == "DTypePolicy":
                policy_name = d_config.get("config", {}).get("name", None)
                if policy_name is not None:
                    config["dtype"] = tf.keras.mixed_precision.Policy(policy_name)
        return cls(**config)