import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, Input
from data import load_dataset, load_unlabeled_dataset
from embedding_utils import images_from_paths
import datetime, os
from model import build_model
from archivist import Archivist


archivist = Archivist()

(X_train, y_train), (X_val, y_val), class_names = load_dataset(
    "./Labeled_Dataset",
    test_split=0.2
)

model = build_model(num_classes=len(class_names))

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=7
    +5,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        )
    ]
)

model.save("image_embedding_model.keras")

# Store embeddings
archivist.store_embeddings(
    model,
    X_train,
    y_train,
    class_names,
    split="train"
)

# Chaos sorting
X_chaos, chaos_paths = load_unlabeled_dataset("./chaos_data")

archivist.sort_chaos_dataset(
    model=model,
    X=X_chaos,
    image_paths=chaos_paths,
    class_names=class_names,
    output_dir="./Sorted_Chaos",
    k=5
)

