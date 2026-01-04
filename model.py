import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, Input, regularizers

def build_model(num_classes, embedding_dim=48):
    inputs = Input(shape=(32, 32, 3))

    # Preprocessing + augmentation
    x = tf.keras.Sequential([
        layers.Rescaling(1.0 / 255.0),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name="preprocess")(inputs)

    # Conv blocks
    for filters in [32, 64]:
        x = layers.Conv2D(
            filters, 3, padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(
        128, 3, padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)

    # Embedding layer
    embedding = layers.Dense(
        embedding_dim,
        use_bias=False,
        kernel_regularizer=regularizers.l2(1e-4),
        name="embedding_dense"
    )(x)
    embedding = layers.BatchNormalization()(embedding)

    # wrap L2 normalization in a Keras layer
    embedding = layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=1),
        name="embedding"
    )(embedding)

    outputs = layers.Dense(num_classes, activation="softmax")(embedding)

    return Model(inputs, outputs)
