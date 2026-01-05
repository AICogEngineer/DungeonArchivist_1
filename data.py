import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
from collections import defaultdict

def load_dataset(
    root_dir,
    img_size=(32, 32),
    test_split=0.2,
    shuffle=True,
    random_state=42,
    verbose=True
):
    if not os.path.exists(root_dir):
        raise ValueError(f"Directory {root_dir} does not exist")

    images = []
    labels = []

    class_names = set()

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

    files = [
        f for f in os.listdir(root_dir)
        if f.lower().endswith(valid_exts)
    ]

    if len(files) == 0:
        raise ValueError(f"No image files found in {root_dir}")

    if verbose:
        print(f"Found {len(files)} images")

    # --- Extract class names from filenames ---
    for fname in files:
        class_name = fname.split("_")[0]
        class_names.add(class_name)

    class_names = sorted(class_names)
    class_to_index = {c: i for i, c in enumerate(class_names)}

    if verbose:
        print(f"Detected {len(class_names)} classes:")
        print(class_names)

    # --- Load images ---
    skipped = 0
    for fname in files:
        path = os.path.join(root_dir, fname)
        class_name = fname.split("_")[0]

        try:
            img = load_img(path, target_size=img_size)
            img = img_to_array(img)
            images.append(img)
            labels.append(class_to_index[class_name])
        except Exception as e:
            skipped += 1
            if verbose:
                print(f"Skipped {fname}: {e}")

    if len(images) == 0:
        raise ValueError("No images could be loaded")

    X = np.array(images, dtype="float32") / 255.0
    y = np.array(labels, dtype="int32")

    if verbose:
        print(f"Loaded {len(X)} images ({skipped} skipped)")
        print(f"X shape: {X.shape}, y shape: {y.shape}")

    if test_split > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_split,
            random_state=random_state,
            shuffle=shuffle,
            stratify= None
        )
        return (X_train, y_train), (X_test, y_test), class_names

    return X, y, class_names


IMG_SIZE = 32

def load_unlabeled_dataset(root_dir):
    image_paths = []

    for fname in os.listdir(root_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            image_paths.append(os.path.join(root_dir, fname))

    images = []

    for path in image_paths:
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.float32) / 255.0
        images.append(img)

    X = tf.stack(images, axis=0)
    return X, image_paths
