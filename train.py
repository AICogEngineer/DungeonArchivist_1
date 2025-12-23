import tensorflow as tf
from model import build_model
from data import load_dataset

(X_train, y_train), (X_val, y_val), class_names = load_dataset(
    "./Labeled_Dataset",
    test_split=0.2
)
model = build_model(num_classes=len(class_names))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=20
)
