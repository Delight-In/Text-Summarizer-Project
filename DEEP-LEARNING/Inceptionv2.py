import tensorflow as tf
from tensorflow.keras import layers, models

def conv2d_bn(x, filters, kernel_size, strides=(1, 1), padding='same', activation='relu'):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x

def inception_module(x, filters):
    # Branch 1: 1x1 convolution
    branch1x1 = conv2d_bn(x, filters[0], (1, 1))

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch3x3 = conv2d_bn(x, filters[1], (1, 1))
    branch3x3 = conv2d_bn(branch3x3, filters[2], (3, 3))

    # Branch 3: 1x1 convolution followed by 5x5 convolution
    branch5x5 = conv2d_bn(x, filters[3], (1, 1))
    branch5x5 = conv2d_bn(branch5x5, filters[4], (5, 5))

    # Branch 4: 3x3 max pooling followed by 1x1 convolution
    branch_pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, filters[5], (1, 1))

    # Concatenate all branches
    outputs = layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1)
    return outputs

def InceptionV2(input_shape=(224, 224, 3), num_classes=1000):
    inputs = layers.Input(shape=input_shape)

    # Initial layers
    x = conv2d_bn(inputs, 32, (3, 3), strides=(2, 2))
    x = conv2d_bn(x, 32, (3, 3))
    x = conv2d_bn(x, 64, (3, 3), padding='same')

    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, (1, 1))
    x = conv2d_bn(x, 192, (3, 3))

    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Inception modules
    x = inception_module(x, [64, 128, 128, 32, 32, 32])
    x = inception_module(x, [64, 128, 128, 32, 32, 32])
    x = inception_module(x, [64, 128, 128, 32, 32, 32])

    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = inception_module(x, [64, 128, 128, 32, 32, 32])
    x = inception_module(x, [64, 128, 128, 32, 32, 32])
    x = inception_module(x, [64, 128, 128, 32, 32, 32])

    # Final layers
    x = layers.AveragePooling2D((7, 7), padding='valid')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# Create the Inception V2 model
model = InceptionV2()
model.summary()
