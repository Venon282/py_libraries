import tensorflow as tf
from tensorflow.keras.layers import Layer # type: ignore

# Custom layer for a weighted average merge
class WeightedAverage(Layer):
    def __init__(self, **kwargs):
        super(WeightedAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        # One scalar weight per input branch.
        self.w = self.add_weight(name='w', shape=(len(input_shape),), initializer='ones', trainable=True)
        super(WeightedAverage, self).build(input_shape)

    def call(self, inputs):
        weighted = [self.w[i] * inputs[i] for i in range(len(inputs))]
        return tf.add_n(weighted) / tf.reduce_sum(self.w)