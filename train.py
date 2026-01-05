import os
import tensorflow as tf
from model import build_model
from data import load_dataset, load_unlabeled_dataset
from archivist import Archivist
import datetime, os

# ---------- LOAD LABELED DATA ----------
(X_train, y_train), (X_val, y_val), class_names = load_dataset("./Labeled_Dataset")

# ---------- BUILD MODEL ----------
model = build_model(num_classes=len(class_names))
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

log_dir = "logs/datadiff/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)

callbacks = [
    #tf.keras.callbacks.TensorBoard(log_dir=log_dir),
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

# ---------- TRAIN MODEL ----------
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=50,  
    callbacks=callbacks
)

model.save("image_embedding_model.keras")

# ---------- STORE EMBEDDINGS ----------
archivist = Archivist()
archivist.store_embeddings(model, X_train, y_train, class_names, split="train")

# ---------- LOAD CHAOS DATA ----------
X_chaos, chaos_paths = load_unlabeled_dataset("./chaos_data")

# ---------- SORT CHAOS IN BATCHES ----------
batch_size = 64
output_dir = "./restored_archive"
for i in range(0, len(X_chaos), batch_size):
    X_batch = X_chaos[i:i+batch_size]
    paths_batch = chaos_paths[i:i+batch_size]
    archivist.sort_chaos_dataset(
        model=model,
        X=X_batch,
        image_paths=paths_batch,
        class_names=class_names,
        output_dir=output_dir,
        k=5
    )
