import tensorflow as tf
import numpy as np

IMG_SIZE = 32


def load_and_preprocess_image(x):
    """
    x can be:
    - a string path
    - a numpy array (already loaded image)
    """

    # Case 1: file path
    if isinstance(x, (str, bytes)):
        img = tf.io.read_file(x)
        img = tf.image.decode_png(img, channels=3)

    # Case 2: numpy array (already loaded)
    elif isinstance(x, np.ndarray):
        img = tf.convert_to_tensor(x)

    else:
        raise TypeError(f"Unsupported input type: {type(x)}")

    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img


def images_from_paths(inputs):
    imgs = [load_and_preprocess_image(x) for x in inputs]
    return tf.stack(imgs, axis=0)
