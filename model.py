from tensorflow.keras import layers, Model, Input

def build_model(num_classes, embedding_dim=64):
    inputs = Input(shape=(32, 32, 3))

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)


    embedding = layers.Dense(
        embedding_dim,
        activation=None,
        name="embedding"
    )(x)

    x = layers.BatchNormalization()(embedding)
    x = layers.ReLU()(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model
