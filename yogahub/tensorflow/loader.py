import io
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from src.tensorflow.aug import test_transform, train_transform


def extract_data(file_path, root="Yoga82/dataset"):
    imgs = []
    labels = []
    with open(file_path, "r") as file:
        for line in file:
            # Split the line on the comma
            parts = line.strip().split(",")
            # Get the image path and labels
            image_path = parts[0]
            image_path = os.path.join(root, image_path)
            label = ",".join(parts[1:])
            if os.path.exists(image_path):
                imgs.append(image_path)
                labels.append([int(word) for word in label.split(",")])
    return imgs, labels


def decode_image(image_path):
    """Decode an image using PIL and return it as a numpy array."""
    image_data = tf.io.read_file(image_path).numpy()
    pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return np.array(pil_image)


def tf_decode_image(image_path):
    """TF wrapper for the decode_image function."""
    return tf.numpy_function(func=decode_image, inp=[image_path], Tout=tf.uint8)


def aug_fn(image, aug):
    """Apply Albumentations augmentations."""
    data = {"image": image}
    if aug == "train_transform":
        aug_data = train_transform(**data)
    else:
        aug_data = test_transform(**data)
    return aug_data["image"]


def tf_aug_fn(image, aug):
    """TF wrapper for the aug_fn function."""
    return tf.numpy_function(func=aug_fn, inp=[image, aug], Tout=tf.float32)


def process_data(image_path, label, depths=[6, 20, 82], aug="train_transform"):
    image = tf_decode_image(image_path)
    aug_img = tf_aug_fn(image, aug)
    labels = tf.split(label, num_or_size_splits=len(depths), axis=0)
    one_hot_labels = [tf.one_hot(l, depth=d) for l, d in zip(labels, depths)]
    combined_label = tf.concat(one_hot_labels, axis=-1)
    return aug_img, combined_label
