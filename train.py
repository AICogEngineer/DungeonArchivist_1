import tensorflow as tf
from model import build_model
from data import load_dataset

X_train, y_train, class_names = load_dataset("./Dungeon_Crawler_Data")

model = build_model(num_classes=len(class_names))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.2
)