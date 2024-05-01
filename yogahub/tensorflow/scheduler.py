import tensorflow as tf


# Define custom learning rate schedule with warmup
class WarmUpSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_steps, total_steps):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = tf.cast(warmup_steps, tf.int64)
        self.total_steps = tf.cast(total_steps, tf.int64)

    def __call__(self, step):
        # step = tf.cast(step, tf.float32)
        warmup_ratio = step / self.warmup_steps
        warmup_lr = self.initial_learning_rate * warmup_ratio
        decay_steps = tf.maximum(self.total_steps - self.warmup_steps, 1)
        decay_ratio = (step - self.warmup_steps) / decay_steps
        decay_lr = self.initial_learning_rate * (1 - decay_ratio)
        return tf.where(step < self.warmup_steps, warmup_lr, decay_lr)
