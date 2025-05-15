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
        
    def plot(self, max_steps: int = 20000, show: bool = False, save: str|None = None):
        """
        compute LR for steps [1..max_steps],
        and plot via matplotlib.
        """
        import matplotlib.pyplot as plt
        
        # Prepare step tensor & compute lrs
        steps = tf.range(1, max_steps + 1, dtype=tf.float32)
        lrs   = self(steps)
        
        if not (save or show):
            return steps, lrs

        # plot
        plt.figure(figsize=(8,4))
        plt.plot(steps.numpy(), lrs.numpy(), linewidth=2)
        plt.xlabel("Training Step")
        plt.ylabel("Learning Rate")
        plt.title(f"Noam Schedule (d_model={self.d_model}, warmup={self.warmup_steps})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        
        if save:
            from pathlib import Path
            file_path = Path(save)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch(exist_ok=True)
            plt.savefig(save)
        
        if show:
            plt.show()
        plt.close()
        
        return steps, lrs