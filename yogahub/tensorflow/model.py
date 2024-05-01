import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras


@keras.saving.register_keras_serializable()
class YogaModel(tf.keras.Model):
    def __init__(self, hub_url, trainable):
        super().__init__()
        self.hub_model = hub.KerasLayer(
            hub_url, trainable=trainable, arguments=dict(return_endpoints=True)
        )
        self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier_head_1 = tf.keras.layers.Dense(
            6, activation="softmax", name="classifier_output_1"
        )
        self.classifier_head_2 = tf.keras.layers.Dense(
            20, activation="softmax", name="classifier_output_2"
        )
        self.classifier_head_3 = tf.keras.layers.Dense(
            82, activation="softmax", name="classifier_output_3"
        )

    def call(self, inputs):
        dict_output = self.hub_model(inputs)
        C3 = dict_output["resnet_v2_50/block2/unit_1/bottleneck_v2/shortcut"]
        C4 = dict_output["resnet_v2_50/block3/unit_1/bottleneck_v2/shortcut"]
        C5 = dict_output["resnet_v2_50/block4/unit_1/bottleneck_v2/shortcut"]

        # Apply Global Average Pooling to each of C3, C4, and C5
        C3_pooled = self.global_avg_pooling(C3)
        C4_pooled = self.global_avg_pooling(C4)
        C5_pooled = self.global_avg_pooling(C5)

        output_1 = self.classifier_head_1(C3_pooled)
        output_2 = self.classifier_head_2(C4_pooled)
        output_3 = self.classifier_head_3(C5_pooled)

        # output = tf.concat([output_1, output_2, output_3], axis=-1)

        return (output_1, output_2, output_3)


# Now, you can pass an input image (or batch of images) to the model:
# outputs = model(input_image)
