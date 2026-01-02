import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, Input, regularizers

def build_model(num_classes, embedding_dim=48):
    inputs = Input(shape=(32, 32, 3))


    # # Data Augmentation
    # x = tf.keras.Sequential([
    #     layers.RandomFlip("horizontal"),
    #     layers.RandomRotation(0.1),
    #     layers.RandomZoom(0.1),
    #     layers.RandomTranslation(0.1, 0.1),
    # ], name="data_augmentation")(inputs)

    # # Normalize Inputs
    
    # x = layers.Rescaling(1./255)(inputs)

    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)


    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)


    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(1e-4), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)


    x = layers.GlobalAveragePooling2D()(x)

    # Embeddings w/ dropout
    x = layers.Dense(embedding_dim, kernel_regularizer=regularizers.l2(1e-4), use_bias=False, name="embedding")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)


    return Model(inputs, outputs)