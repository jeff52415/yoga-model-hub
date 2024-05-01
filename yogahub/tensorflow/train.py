import os
import warnings

import tensorflow as tf
from loguru import logger
from tensorflow import keras
from tqdm import tqdm

from yogahub.cfg.train import classify_config as config
from yogahub.tensorflow.loader import extract_data, process_data
from yogahub.tensorflow.loss_metric import Accuracy, SmoothedCategoricalCrossentropy
from yogahub.tensorflow.model import YogaModel
from yogahub.tensorflow.scheduler import WarmUpSchedule

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = config.device

"""
mirrored_strategy = tf.distribute.MirroredStrategy(
    devices=tf.config.list_logical_devices("GPU")
)
"""
# tf.distribute.experimental_set_strategy(mirrored_strategy)


# device
gpu_number = len(tf.config.list_physical_devices("GPU"))
logger.info(f"Num GPUs Available: {gpu_number}")
logger.info(f"TensorFlow version: {tf.__version__}")
AUTOTUNE = tf.data.experimental.AUTOTUNE


# set up model
hub_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
trainable = True
# Instantiate the custom model
if config.pretrained:
    # https://www.tensorflow.org/tutorials/keras/save_and_load
    model = tf.keras.models.load_model(config.pretrained)
    logger.info(f"Successfully load pretrained: {config.pretrained}")
else:
    model = YogaModel(hub_url, trainable)
    # model.build(input_shape=([None, 440, 440, 3]))
    # model.load_weights(config.pretrained)
    logger.info("Initialize model from scratch")
model.summary()


# load data
train_img, train_label = extract_data(config.train_path)
test_img, test_label = extract_data(config.test_path)
BUFFER_SIZE = len(train_img)


def prepare_dataset(
    img, label, batch_size, shuffle_buffer_size=None, aug="train_transform"
):
    dataset = tf.data.Dataset.from_tensor_slices((img, label))
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(
        lambda x, y: process_data(x, y, [6, 20, 82], aug),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


train_data = prepare_dataset(
    train_img, train_label, config.batch_size, shuffle_buffer_size=BUFFER_SIZE
)
test_data = prepare_dataset(
    test_img, test_label, config.batch_size, aug="test_transform"
)


# loss & metric
metric_6 = Accuracy()
metric_20 = Accuracy()
metric_82 = Accuracy()
metrics = [metric_6, metric_20, metric_82]

loss = SmoothedCategoricalCrossentropy()
train_loss_recorder = tf.keras.metrics.Mean(name="train_loss")
test_loss_recorder = tf.keras.metrics.Mean(name="test_loss")


# optimizer and scheduler
num_steps = len(train_data) * config.num_epochs
warmup_period = int(config.warmup_period * num_steps)

lr_schedule = WarmUpSchedule(
    initial_learning_rate=config.lr, warmup_steps=warmup_period, total_steps=num_steps
)
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=lr_schedule, weight_decay=config.weight_decay
)


# train & test
@tf.function
def train_step(images, labels, metrics):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        labels_split = tf.split(labels, [6, 20, 82], axis=-1)
        labels_split = [tf.squeeze(label, axis=1) for label in labels_split]
        losses = []
        for label, predict, metric in zip(labels_split, predictions, metrics):
            losses.append(loss(label, predict))
            metric.update_state(label, predict)

        total_loss = tf.reduce_sum(losses) / len(losses)
    train_loss_recorder(total_loss)
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


@tf.function
def test_step(images, labels, metrics):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    labels = tf.split(labels, [6, 20, 82], axis=-1)
    labels = [tf.squeeze(label, axis=1) for label in labels]
    losses = []
    for label, predict, metric in zip(labels, predictions, metrics):
        losses.append(loss(label, predict))
        metric.update_state(label, predict)
    total_loss = tf.reduce_sum(losses) / len(losses)
    test_loss_recorder(total_loss)


EPOCHS = config.num_epochs
best_accuracy = 0
for epoch in range(EPOCHS):
    logger.info(f"Epoch {epoch+1}/{EPOCHS}")

    # Reset the metrics at the start of the next epoch
    train_loss_recorder.reset_states()
    test_loss_recorder.reset_states()
    metric_6.reset_states()
    metric_20.reset_states()
    metric_82.reset_states()

    for images, labels in tqdm(train_data, total=len(train_data), desc="Training"):
        train_step(images, labels, metrics)

    metric_6.reset_states()
    metric_20.reset_states()
    metric_82.reset_states()

    for images, labels in tqdm(test_data, total=len(test_data), desc="Testing"):
        test_step(images, labels, metrics)

    metrics_result = [metric.result().get("acc").numpy() for metric in metrics]
    logger.info(
        f"Epoch {epoch+1}/{EPOCHS}, "
        f"Train Loss: {train_loss_recorder.result()}, "
        f"Test Loss: {test_loss_recorder.result()}, "
        f"Metrics: {metrics_result}"
    )

    current_accuracy = metrics_result[-1]
    if best_accuracy < current_accuracy:
        model.save("yoga_check.keras", save_format="keras")
        best_accuracy = current_accuracy
