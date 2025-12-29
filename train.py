import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, Input
from data import load_dataset
import datetime, os
from model import build_model

(X_train, y_train), (X_val, y_val), class_names = load_dataset(
    "./Labeled_Dataset",
    test_split=0.2
)

model = build_model(num_classes=len(class_names))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

log_dir = "logs/datadiff/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1
    )
]

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=50,
    callbacks=callbacks
)