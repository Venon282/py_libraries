import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class Noam(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super(Noam, self).__init__(**kwargs)
        # Store d_model as an integer for serialization
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        d_model = tf.cast(self.d_model, tf.float32)
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
        }
            