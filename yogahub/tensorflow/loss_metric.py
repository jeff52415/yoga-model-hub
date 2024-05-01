import tensorflow as tf


# Define custom loss with label smoothing
class SmoothedCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing
        # https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy
        self.loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=self.label_smoothing,
            reduction=tf.keras.losses.Reduction.NONE,
        )

    def call(self, y_true, y_pred):
        loss = self.loss(y_true, y_pred)
        return loss


class Accuracy(tf.keras.metrics.Metric):
    def __init__(self, name="custom_accuracy", **kwargs):
        super(Accuracy, self).__init__(name=name, **kwargs)
        # https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalAccuracy
        self.acc = tf.keras.metrics.CategoricalAccuracy()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.acc.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return {"acc": self.acc.result()}

    def reset_states(self):
        self.acc.reset_states()
