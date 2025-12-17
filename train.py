

def sprite_encoder():
    '''
    EXAMPLE AUTOENCODER

    input_img = keras.Input(shape=(32, 32, 3))

    img = layers.Conv2D(32, (3, 3), activation='relu', padding = 'same')(input_img)
    img = layers.MaxPooling2D((2, 2), padding='same')(img)
        #   output: (15, 15, 32)
    img = layers.Conv2D(64, (3, 3), activation='relu', padding = 'same')(img)
    img = layers.MaxPooling2D((2, 2), padding='same')(img)
        #   output: (6, 6, 64)
    img = layers.Conv2D(128 (3, 3), activation='relu', padding = 'same')(img)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(img)

    img = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    img = layers.UpSampling2D((2, 2))(img)
    img = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(img)
    img = layers.UpSampling2D((2, 2))(img)
    img = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(img)
    img = layers.UpSampling2D((2, 2))(img)

    decoded = img = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(img)

    autoencoder = models.Model(input_img, decoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return encoder, autoencoder
    '''

    # Things to try:
    #   dropout
    #   batch normalization
    #   more/less layers
    #   different optimizer
    pass
