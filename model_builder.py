from tensorflow.keras import layers, models

def build_conv_pool_model(input_size, length):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_size, padding="valid"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (2, 2), activation='relu', padding="valid"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (2, 2), activation='relu', padding="valid"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(length, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_conv_pool_3d_model(input_size, length):
    model = models.Sequential()
    model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_size, padding="valid"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D((1, 2, 2)))

    model.add(layers.Conv3D(64, (2, 2, 2), activation='relu', padding="valid"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D((1, 2, 2)))

    model.add(layers.Conv3D(128, (2, 2, 2), activation='relu', padding="valid"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D((1, 2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(length, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model