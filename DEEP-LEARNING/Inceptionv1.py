import tensorflow as tf
from tensorflow.keras import layers, models

def inception_module(x, filters):
    # Branch 1: 1x1 convolution
    branch1x1 = layers.Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch3x3 = layers.Conv2D(filters[1], (1, 1), padding='same', activation='relu')(x)
    branch3x3 = layers.Conv2D(filters[2], (3, 3), padding='same', activation='relu')(branch3x3)

    # Branch 3: 1x1 convolution followed by 5x5 convolution
    branch5x5 = layers.Conv2D(filters[3], (1, 1), padding='same', activation='relu')(x)
    branch5x5 = layers.Conv2D(filters[4], (5, 5), padding='same', activation='relu')(branch5x5)

    # Branch 4: 3x3 max pooling followed by 1x1 convolution
    branch_pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = layers.Conv2D(filters[5], (1, 1), padding='same', activation='relu')(branch_pool)

    # Concatenate all branches
    outputs = layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1)
    return outputs

def InceptionV1(input_shape=(224, 224, 3), num_classes=1000):
    inputs = layers.Input(shape=input_shape)

    # Initial convolutional layers
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Inception blocks
    x = inception_module(x, [64, 128, 128, 32, 32, 32])
    x = inception_module(x, [64, 128, 128, 32, 32, 32])
    x = inception_module(x, [64, 128, 128, 32, 32, 32])
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, [64, 128, 128, 32, 32, 32])
    x = inception_module(x, [64, 128, 128, 32, 32, 32])
    x = inception_module(x, [64, 128, 128, 32, 32, 32])
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Final layers
    x = layers.AveragePooling2D((7, 7), padding='valid')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# Create the Inception V1 model
model = InceptionV1()
model.summary()
